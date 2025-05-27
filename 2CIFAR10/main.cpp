#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

// g++ -O2 -std=c++17 main2.cpp -o nn
// ./nn

using namespace std;

// Add these functions to replace the MNIST reading functions

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
        cout << "Loading " << filename << "..." << endl;
        
        CifarData batch_data = read_cifar_batch(filename);
        
        // Append to combined data
        all_images.insert(all_images.end(), 
                         batch_data.images.begin(), 
                         batch_data.images.end());
        all_labels.insert(all_labels.end(),
                         batch_data.labels.begin(),
                         batch_data.labels.end());
    }
    
    cout << "Loaded " << all_images.size() << " training images" << endl;
    return make_pair(all_images, all_labels);
}

pair<vector<vector<float>>, vector<uint8_t>> load_cifar_test() {
    string filename = "cifar-10-batches-bin/test_batch.bin";
    cout << "Loading " << filename << "..." << endl;
    
    CifarData test_data = read_cifar_batch(filename);
    
    cout << "Loaded " << test_data.images.size() << " test images" << endl;
    return make_pair(test_data.images, test_data.labels);
}

// CIFAR-10 class names for reference
vector<string> cifar_classes = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

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

// Define function pointer types
using ActivationFunc = function<float(float)>;
using DerivativeFunc = function<float(float)>;

struct DenseLayer { // Fully connected layer
    int in_size, out_size;
    vector<vector<float>> weights;
    vector<float> biases;
    vector<float> input, output, delta;
    vector<float> pre_activation; // Store values before activation for derivative calculation

    DenseLayer(int in_sz, int out_sz) : in_size(in_sz), out_size(out_sz) {
        weights.resize(out_size, vector<float>(in_size));
        biases.resize(out_size);
        pre_activation.resize(out_size);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &row : weights)
            for (auto &w : row)
                w = dist(gen);
    }

    vector<float> forward(const vector<float>& x, ActivationFunc activation_func = nullptr) {
        input = x;
        output.resize(out_size);
        pre_activation.resize(out_size);
        
        for (int i = 0; i < out_size; ++i) {
            float sum = biases[i];
            for (int j = 0; j < in_size; ++j)
                sum += weights[i][j] * x[j];
            
            pre_activation[i] = sum; // Store pre-activation value
            
            if (activation_func) {
                output[i] = activation_func(sum);
            } else {
                output[i] = sum; // No activation (linear)
            }
        }
        return output;
    }
    
    vector<float> backward(const vector<float>& grad_out, float lr, DerivativeFunc derivative_func = nullptr) {
        delta.assign(in_size, 0.0f);
        for (int i = 0; i < out_size; ++i) {
            float activation_grad = 1.0f; // Default for linear/no activation
            
            if (derivative_func) {
                activation_grad = derivative_func(pre_activation[i]);
            }
            
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

void save_layer(const DenseLayer& L, const string& file) {
    ofstream f(file, ios::binary);
    int in = L.in_size, out = L.out_size;
    f.write((char*)&in,  sizeof(int));
    f.write((char*)&out, sizeof(int));
    for (auto& row : L.weights)  f.write((char*)row.data(), row.size()*sizeof(float));
    f.write((char*)L.biases.data(), L.biases.size()*sizeof(float));
}

int main() {
    // Load CIFAR-10 data
    auto [train_images, train_labels] = load_cifar_train();  // 50,000 samples
    auto [test_images, test_labels] = load_cifar_test();     // 10,000 samples
    
    int n_train = train_images.size();
    int n_test = test_images.size();
    
    cout << "Training samples: " << n_train << endl;
    cout << "Test samples: " << n_test << endl;
    cout << "Input size: " << train_images[0].size() << endl; // Should be 3072
    
    // Network architecture - need bigger network for CIFAR-10
    DenseLayer l1(3072, 512);  // Much larger first layer
    DenseLayer l2(512, 128);   // Add hidden layer
    DenseLayer l3(128, 10);    // Output layer
    
    float lr = 0.001f;  // Lower learning rate for stability
    ActivationFunc activation_func = leaky_relu_activation;
    DerivativeFunc derivative_func = leaky_relu_derivative;

    // Open result file for writing
    ofstream result_file("cifar_results.txt");
    
    // Training loop
    for (int epoch = 0; epoch < 20; epoch++) {  // More epochs needed
        float total_loss = 0.0f;
        int correct = 0;
        
        // Shuffle training data each epoch
        vector<int> indices(n_train);
        iota(indices.begin(), indices.end(), 0);
        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);
        
        for (int idx = 0; idx < n_train; idx++) {
            int i = indices[idx];
            auto x = train_images[i];
            int y = train_labels[i];

            // Forward pass through all layers
            auto h1 = l1.forward(x, activation_func);
            auto h2 = l2.forward(h1, activation_func);
            auto out = l3.forward(h2); // No activation for output layer
            auto pred = softmax(out);
            
            total_loss += cross_entropy(pred, y);
            correct += (argmax(pred) == y);

            // Backward pass
            auto grad = softmax_loss_backward(pred, y);
            auto grad_l3 = l3.backward(grad, lr);
            auto grad_l2 = l2.backward(grad_l3, lr, derivative_func);
            l1.backward(grad_l2, lr, derivative_func);
        }
        
        string epoch_msg = "Epoch " + to_string(epoch + 1) + 
                          " - Loss: " + to_string(total_loss / n_train) + 
                          ", Accuracy: " + to_string((float)correct / n_train * 100) + "%\n";
        
        cout << epoch_msg;
        result_file << epoch_msg;
        
        // Evaluate on test set every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            int test_correct = 0;
            for (int i = 0; i < n_test; ++i) {
                auto h1 = l1.forward(test_images[i], activation_func);
                auto h2 = l2.forward(h1, activation_func);
                auto out = l3.forward(h2);
                auto pred = softmax(out);
                if (argmax(pred) == test_labels[i])
                    test_correct++;
            }
            
            string test_msg = "  Test Accuracy: " + to_string((float)test_correct / n_test * 100) + "%\n";
            cout << test_msg;
            result_file << test_msg;
        }
    }
    
    // Final test evaluation
    int correct = 0;
    for (int i = 0; i < n_test; ++i) {
        auto h1 = l1.forward(test_images[i], activation_func);
        auto h2 = l2.forward(h1, activation_func);
        auto out = l3.forward(h2);
        auto pred = softmax(out);
        if (argmax(pred) == test_labels[i])
            correct++;
    }
    
    string final_msg = "Final Test Accuracy: " + to_string((float)correct / n_test * 100) + "%\n";
    cout << final_msg;
    result_file << final_msg;
    
    result_file.close();
    
    // Save the trained models
    save_layer(l1, "cifar_l1.bin");
    save_layer(l2, "cifar_l2.bin");
    save_layer(l3, "cifar_l3.bin");
    
    return 0;
}