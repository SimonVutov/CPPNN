#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

// g++ -O2 -std=c++17 main.cpp -o nn
// ./nn

using namespace std;

void msg(const string& message) {
    // open in append mode
    ofstream result_file("result.txt", ios::app);
    if (!result_file) {
        cerr << "Could not open result.txt for appending\n";
        return;
    }

    // write to console
    cout << message;

    // append to file
    result_file << message;
    // no need to explicitly close – it'll close when it goes out of scope
}

struct CifarData {
    vector<vector<float>> images;
    vector<uint8_t> labels;
};

CifarData read_cifar_batch(const string& filename) {
    ifstream file(filename, ios::binary);
    assert(file.is_open());
    
    CifarData data;
    
    // Each CIFAR-10 file contains 10,000 records
    // Each record: 1 byte label + 3072 bytes image data (32*32*3)
    const int num_samples = 10000;
    const int image_size = 32 * 32 * 3; // 3072
    
    data.images.resize(num_samples, vector<float>(image_size));
    data.labels.resize(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        // Read label (1 byte)
        unsigned char label;
        file.read((char*)&label, 1);
        data.labels[i] = label;
        
        // Read image data (3072 bytes)
        // Format: 1024 red bytes, 1024 green bytes, 1024 blue bytes
        vector<unsigned char> raw_image(image_size);
        file.read((char*)raw_image.data(), image_size);
        
        // Convert to float and normalize [0, 255] -> [0, 1]
        for (int j = 0; j < image_size; ++j) {
            data.images[i][j] = raw_image[j] / 255.0f;
        }
    }
    
    file.close();
    return data;
}

pair<vector<vector<float>>, vector<uint8_t>> load_cifar_train() {
    vector<vector<float>> all_images;
    vector<uint8_t> all_labels;
    
    // Load all 5 training batches
    for (int batch = 1; batch <= 5; ++batch) {
        string filename = "cifar-10-batches-bin/data_batch_" + to_string(batch) + ".bin";
        msg("Loading " + filename + "...\n");
        
        CifarData batch_data = read_cifar_batch(filename);
        
        // Append to combined data
        all_images.insert(all_images.end(), 
                         batch_data.images.begin(), 
                         batch_data.images.end());
        all_labels.insert(all_labels.end(),
                         batch_data.labels.begin(),
                         batch_data.labels.end());
    }
    
    msg("Loaded " + to_string(all_images.size()) + " training images\n");
    return make_pair(all_images, all_labels);
}

pair<vector<vector<float>>, vector<uint8_t>> load_cifar_test() {
    string filename = "cifar-10-batches-bin/test_batch.bin";
    msg("Loading " + filename + "...\n");
    
    CifarData test_data = read_cifar_batch(filename);
    
    msg("Loaded " + to_string(test_data.images.size()) + " test images\n");
    return make_pair(test_data.images, test_data.labels);
}

// Activation functions
float relu(float x) {
    return max(0.0f, x);
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

float tanh_activation(float x) {
    return tanh(x);
}

float tanh_derivative(float x) {
    float t = tanh(x);
    return 1.0f - t * t;
}

float sigmoid_activation(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid_activation(x);
    return s * (1.0f - s);
}

// Linear activation (no activation)
float linear_activation(float x) {
    return x;
}

float linear_derivative(float x) {
    return 1.0f;
}

// Leaky ReLU activation
float leaky_relu_activation(float x) {
    return max(0.01f * x, x);
}
float leaky_relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.01f;
}

// CIFAR-10 class names for reference
vector<string> cifar_classes = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Define function pointer types
using ActivationFunc = function<float(float)>;
using DerivativeFunc = function<float(float)>;

// Activation type enum for easy specification
enum class ActivationType {
    LINEAR,
    RELU,
    LEAKY_RELU,
    TANH,
    SIGMOID
};

// Helper function to get activation functions
pair<ActivationFunc, DerivativeFunc> get_activation_functions(ActivationType type) {
    switch (type) {
        case ActivationType::RELU:
            return {relu, relu_derivative};
        case ActivationType::LEAKY_RELU:
            return {leaky_relu_activation, leaky_relu_derivative};
        case ActivationType::TANH:
            return {tanh_activation, tanh_derivative};
        case ActivationType::SIGMOID:
            return {sigmoid_activation, sigmoid_derivative};
        case ActivationType::LINEAR:
        default:
            return {linear_activation, linear_derivative};
    }
}

struct DenseLayer { // Fully connected layer
    int in_size, out_size;
    vector<vector<float>> weights;
    vector<float> biases;
    vector<float> input, output, delta;
    vector<float> pre_activation; // Store values before activation for derivative calculation
    ActivationType activation_type;
    ActivationFunc activation_func;
    DerivativeFunc derivative_func;

    DenseLayer(int in_sz, int out_sz, ActivationType act_type = ActivationType::LINEAR) 
        : in_size(in_sz), out_size(out_sz), activation_type(act_type) {
        weights.resize(out_size, vector<float>(in_size));
        biases.resize(out_size);
        pre_activation.resize(out_size);
        output.resize(out_size);
        
        // Set activation functions
        auto [act_func, deriv_func] = get_activation_functions(activation_type);
        activation_func = act_func;
        derivative_func = deriv_func;
        
        // Xavier/Glorot initialization
        random_device rd;
        mt19937 gen(rd());
        float limit = sqrt(6.0f / (in_size + out_size));
        uniform_real_distribution<float> dist(-limit, limit);
        
        for (auto &row : weights)
            for (auto &w : row)
                w = dist(gen);
        
        // Initialize biases to zero
        fill(biases.begin(), biases.end(), 0.0f);
    }

    vector<float> forward(const vector<float>& x) {
        input = x;
        
        for (int i = 0; i < out_size; ++i) {
            float sum = biases[i];
            for (int j = 0; j < in_size; ++j)
                sum += weights[i][j] * x[j];
            
            pre_activation[i] = sum;
            output[i] = activation_func(sum);
        }
        return output;
    }
    
    vector<float> backward(const vector<float>& grad_out, float lr) {
        delta.assign(in_size, 0.0f);
        
        for (int i = 0; i < out_size; ++i) {
            float activation_grad = derivative_func(pre_activation[i]);
            float grad = grad_out[i] * activation_grad;
            
            for (int j = 0; j < in_size; ++j) {
                delta[j] += grad * weights[i][j];
                weights[i][j] -= lr * grad * input[j];
            }
            biases[i] -= lr * grad;
        }
        return delta;
    }
};

// Softmax + Cross-Entropy loss
vector<float> softmax(const vector<float> &logits) {
  float max_logit = *max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  vector<float> probs(logits.size());
  for (int i = 0; i < logits.size(); ++i) {
    probs[i] = exp(logits[i] - max_logit);
    sum += probs[i];
  }
  for (float &p : probs)
    p /= sum;
  return probs;
}

float cross_entropy(const vector<float> &pred, int label) {
  return -log(pred[label] + 1e-8f);
}

vector<float> softmax_loss_backward(const vector<float> &pred, int label) {
  vector<float> grad = pred;
  grad[label] -= 1.0f;
  return grad;
}

int argmax(const vector<float> &v) {
  return max_element(v.begin(), v.end()) - v.begin();
}

struct Network {
    vector<DenseLayer> layers;
    vector<vector<float>> activations; // Store activations for each layer
    float lr;

    Network(float learning_rate = 0.001f) : lr(learning_rate) {}

    // Add dense layer to the network
    void add_dense(int input_size, int output_size, ActivationType activation = ActivationType::RELU) {
        layers.emplace_back(input_size, output_size, activation);
        activations.resize(layers.size());
    }

    // Build network from layer sizes and activations
    void build(const vector<int>& layer_sizes, 
               const vector<ActivationType>& activations_types = {}) {
        layers.clear();
        
        for (int i = 0; i < layer_sizes.size() - 1; ++i) {
            ActivationType act_type = ActivationType::RELU; // default
            
            // Use provided activation or default
            if (i < activations_types.size()) {
                act_type = activations_types[i];
            }
            
            // Last layer typically uses linear activation for classification
            if (i == layer_sizes.size() - 2 && activations_types.empty()) {
                act_type = ActivationType::LINEAR;
            }
            
            add_dense(layer_sizes[i], layer_sizes[i + 1], act_type);
        }
        
        activations.resize(layers.size());
        msg("Network built with " + to_string(layers.size()) + " layers:\n");
        for (int i = 0; i < layers.size(); ++i) {
            msg("  Layer " + to_string(i + 1) + ": " + to_string(layers[i].in_size) 
                 + " -> " + to_string(layers[i].out_size) + " (");
            
            switch (layers[i].activation_type) {
                case ActivationType::RELU: msg("ReLU"); break;
                case ActivationType::LEAKY_RELU: msg("Leaky ReLU"); break;
                case ActivationType::TANH: msg("Tanh"); break;
                case ActivationType::SIGMOID: msg("Sigmoid"); break;
                case ActivationType::LINEAR: msg("Linear"); break;
            }
            msg(")\n");
        }
    }

    // Forward pass through the entire network
    vector<float> forward(const vector<float>& x) {
        vector<float> current_input = x;
        
        for (int i = 0; i < layers.size(); ++i) {
            activations[i] = layers[i].forward(current_input);
            current_input = activations[i];
        }
        
        return activations.back(); // Return final output
    }

    // Backward pass through the entire network
    void backward(const vector<float>& grad_output) {
        vector<float> current_grad = grad_output;
        
        for (int i = layers.size() - 1; i >= 0; --i) {
            current_grad = layers[i].backward(current_grad, lr);
        }
    }

    // Train on a single sample
    float train_sample(const vector<float>& x, int y) {
        // Forward pass
        auto output = forward(x);
        auto pred = softmax(output);
        
        // Calculate loss
        float loss = cross_entropy(pred, y);
        
        // Backward pass
        auto grad = softmax_loss_backward(pred, y);
        backward(grad);
        
        return loss;
    }

    // Predict single sample
    int predict(const vector<float>& x) {
        auto output = forward(x);
        auto pred = softmax(output);
        return argmax(pred);
    }

    // Evaluate accuracy on dataset
    float evaluate(const vector<vector<float>>& X, const vector<uint8_t>& y) {
        int correct = 0;
        for (int i = 0; i < X.size(); ++i) {
            if (predict(X[i]) == y[i]) {
                correct++;
            }
        }
        return (float)correct / X.size();
    }

    // Train the network
    void train(const vector<vector<float>>& X_train, const vector<uint8_t>& y_train,
               const vector<vector<float>>& X_test, const vector<uint8_t>& y_test,
               int epochs = 10, bool shuffle_data = true, int eval_every = 5) {
        
        int n_train = X_train.size();
        vector<int> indices(n_train);
        iota(indices.begin(), indices.end(), 0);
        
        random_device rd;
        mt19937 g(rd());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            
            // Shuffle training data
            if (shuffle_data) {
                shuffle(indices.begin(), indices.end(), g);
            }
            
            // Train on all samples
            for (int idx = 0; idx < n_train; ++idx) {
                int i = indices[idx];
                total_loss += train_sample(X_train[i], y_train[i]);
            }
            
            // Calculate training accuracy
            float train_acc = evaluate(X_train, y_train);
            
            msg("Epoch " + to_string(epoch + 1) + "/" + to_string(epochs) 
                 + " - Loss: " + to_string(total_loss / n_train)
                 + " - Train Acc: " + to_string(train_acc * 100) + "%\n");
            
            // Evaluate on test set
            if ((epoch + 1) % eval_every == 0 || epoch == epochs - 1) {
                float test_acc = evaluate(X_test, y_test);
                msg("  Test Accuracy: " + to_string(test_acc * 100) + "%\n");
            }
        }
    }

    // Set learning rate
    void set_learning_rate(float new_lr) {
        lr = new_lr;
    }

    // Save network to files
    void save(const string& prefix) {
        for (int i = 0; i < layers.size(); ++i) {
            string filename = prefix + "_layer_" + to_string(i) + ".bin";
            save_layer(layers[i], filename);
        }
        msg("Network saved with prefix: " + prefix + "\n");
    }

private:
    void save_layer(const DenseLayer& L, const string& file) {
        ofstream f(file, ios::binary);
        int in = L.in_size, out = L.out_size;
        int act_type = static_cast<int>(L.activation_type);
        
        f.write((char*)&in, sizeof(int));
        f.write((char*)&out, sizeof(int));
        f.write((char*)&act_type, sizeof(int));
        
        for (auto& row : L.weights) 
            f.write((char*)row.data(), row.size() * sizeof(float));
        f.write((char*)L.biases.data(), L.biases.size() * sizeof(float));
        
        f.close();
    }
};

int main() {
    // Load CIFAR-10 data
    auto [train_images, train_labels] = load_cifar_train();
    auto [test_images, test_labels] = load_cifar_test();
    msg("Training samples: " + to_string(train_images.size()) + "\n" + "Test samples: " + to_string(test_images.size()) + "\n" + "Input size: " + to_string(train_images[0].size()) + "\n");
    
    // Create network using the modular approach
    Network network(0.001f); // learning rate
    
    // Method 1: Build network with layer sizes and activation types
    vector<int> layer_sizes = {3072, 12, 12, 10};
    vector<ActivationType> activations = {
        ActivationType::LEAKY_RELU,  // First hidden layer
        ActivationType::LEAKY_RELU,  // Second hidden layer
        ActivationType::LINEAR       // Output layer
    };
    
    network.build(layer_sizes, activations);
    
    // Train the network
    network.train(train_images, train_labels, test_images, test_labels, 
                  6, true, 5); // 20 epochs, shuffle data, evaluate every 5 epochs
    
    // Save the trained network
    network.save("cifar_network");
    
    return 0;
}