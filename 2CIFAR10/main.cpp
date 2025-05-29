#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <functional>
#include <variant>

// g++ -O2 -std=c++17 main.cpp -o nn && ./nn

using namespace std;

void msg(const string& message) {
    // add time and date to the message
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    string time_str = ctime(&now_time);
    time_str.pop_back(); // remove newline
    const string newMessage = time_str + " : " + message;

    // open in append mode
    ofstream result_file("result.txt", ios::app);
    if (!result_file) {
        cerr << "Could not open result.txt for appending\n";
        return;
    }

    // write to console
    cout << newMessage;

    // append to file
    result_file << newMessage;
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

struct ConvolutionalLayer {
    int in_h, in_w, in_c;
    int out_h, out_w, out_c;
    int kernel_size;
    int stride;
    int padding;
    vector<vector<vector<vector<float>>>> weights; // [out_c][in_c][kernel_size][kernel_size]
    vector<float> biases;
    // Store input and output for backward
    vector<vector<vector<float>>> last_input;
    vector<vector<vector<float>>> last_output;

    ConvolutionalLayer(int in_h, int in_w, int in_c, int out_h, int out_w, int out_c, int kernel_size, int stride, int padding) 
        : in_h(in_h), in_w(in_w), in_c(in_c), out_h(out_h), out_w(out_w), out_c(out_c), kernel_size(kernel_size), stride(stride), padding(padding) {
        weights.resize(out_c, vector<vector<vector<float>>>(in_c, vector<vector<float>>(kernel_size, vector<float>(kernel_size))));
        biases.resize(out_c);
        // Xavier/Glorot initialization
        random_device rd;
        mt19937 gen(rd());
        float fan_in = in_c * kernel_size * kernel_size;
        float fan_out = out_c;
        float limit = sqrt(6.0f / (fan_in + fan_out));
        uniform_real_distribution<float> dist(-limit, limit);
        for (int i = 0; i < out_c; ++i) {
            for (int j = 0; j < in_c; ++j) {
                for (int k = 0; k < kernel_size; ++k) {
                    for (int l = 0; l < kernel_size; ++l) {
                        weights[i][j][k][l] = dist(gen);
                    }
                }
            }
        }
        fill(biases.begin(), biases.end(), 0.0f);
    }

    vector<vector<vector<float>>> pad_input(const vector<vector<vector<float>>>& x) {
        int padded_h = in_h + 2 * padding;
        int padded_w = in_w + 2 * padding;
        vector<vector<vector<float>>> padded(in_c, vector<vector<float>>(padded_h, vector<float>(padded_w, 0.0f)));
        for (int c = 0; c < in_c; ++c) {
            for (int i = 0; i < in_h; ++i) {
                for (int j = 0; j < in_w; ++j) {
                    padded[c][i + padding][j + padding] = x[c][i][j];
                }
            }
        }
        return padded;
    }

    vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>& x) {
        last_input = x;
        auto padded_input = pad_input(x);
        vector<vector<vector<float>>> out(out_c, vector<vector<float>>(out_h, vector<float>(out_w, 0.0f)));
        for (int f = 0; f < out_c; ++f) {
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    float sum = biases[f];
                    for (int c = 0; c < in_c; ++c) {
                        for (int ki = 0; ki < kernel_size; ++ki) {
                            for (int kj = 0; kj < kernel_size; ++kj) {
                                int in_i = i * stride + ki;
                                int in_j = j * stride + kj;
                                sum += padded_input[c][in_i][in_j] * weights[f][c][ki][kj];
                            }
                        }
                    }
                    out[f][i][j] = sum;
                }
            }
        }
        last_output = out;
        return out;
    }

    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>>& grad_out, float lr) {
        // grad_out: [out_c][out_h][out_w]
        // Returns grad_input: [in_c][in_h][in_w]
        auto padded_input = pad_input(last_input);
        int padded_h = in_h + 2 * padding;
        int padded_w = in_w + 2 * padding;
        // Gradients
        vector<vector<vector<vector<float>>>> grad_weights(out_c, vector<vector<vector<float>>>(in_c, vector<vector<float>>(kernel_size, vector<float>(kernel_size, 0.0f))));
        vector<float> grad_biases(out_c, 0.0f);
        vector<vector<vector<float>>> grad_input_padded(in_c, vector<vector<float>>(padded_h, vector<float>(padded_w, 0.0f)));
        // Compute gradients
        for (int f = 0; f < out_c; ++f) {
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    float grad = grad_out[f][i][j];
                    grad_biases[f] += grad;
                    for (int c = 0; c < in_c; ++c) {
                        for (int ki = 0; ki < kernel_size; ++ki) {
                            for (int kj = 0; kj < kernel_size; ++kj) {
                                int in_i = i * stride + ki;
                                int in_j = j * stride + kj;
                                grad_weights[f][c][ki][kj] += grad * padded_input[c][in_i][in_j];
                                grad_input_padded[c][in_i][in_j] += grad * weights[f][c][ki][kj];
                            }
                        }
                    }
                }
            }
        }
        // Update weights and biases
        for (int f = 0; f < out_c; ++f) {
            for (int c = 0; c < in_c; ++c) {
                for (int ki = 0; ki < kernel_size; ++ki) {
                    for (int kj = 0; kj < kernel_size; ++kj) {
                        weights[f][c][ki][kj] -= lr * grad_weights[f][c][ki][kj];
                    }
                }
            }
            biases[f] -= lr * grad_biases[f];
        }
        // Remove padding from grad_input_padded
        vector<vector<vector<float>>> grad_input(in_c, vector<vector<float>>(in_h, vector<float>(in_w, 0.0f)));
        for (int c = 0; c < in_c; ++c) {
            for (int i = 0; i < in_h; ++i) {
                for (int j = 0; j < in_w; ++j) {
                    grad_input[c][i][j] = grad_input_padded[c][i + padding][j + padding];
                }
            }
        }
        return grad_input;
    }
};

// Utility: flatten 3D tensor to 1D vector
vector<float> flatten3D(const vector<vector<vector<float>>>& x) {
    vector<float> out;
    for (const auto& mat : x)
        for (const auto& row : mat)
            for (float v : row)
                out.push_back(v);
    return out;
}
// Utility: reshape 1D vector to 3D tensor (channels, height, width)
vector<vector<vector<float>>> reshape1Dto3D(const vector<float>& x, int c, int h, int w) {
    vector<vector<vector<float>>> out(c, vector<vector<float>>(h, vector<float>(w)));
    int idx = 0;
    for (int ch = 0; ch < c; ++ch)
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                out[ch][i][j] = x[idx++];
    return out;
}

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

// Layer type enum
enum class LayerType { DENSE, CONV, FLATTEN };

struct Layer {
    LayerType type;
    DenseLayer dense;
    ConvolutionalLayer conv;
    // For flatten, no parameters needed
    Layer(DenseLayer d) : type(LayerType::DENSE), dense(std::move(d)), conv(0,0,0,0,0,0,0,0,0) {}
    Layer(ConvolutionalLayer c) : type(LayerType::CONV), dense(0,0), conv(std::move(c)) {}
    Layer() : type(LayerType::FLATTEN), dense(0,0), conv(0,0,0,0,0,0,0,0,0) {}
};

struct Network {
    vector<Layer> layers;
    vector<vector<float>> activations_1d; // for dense
    vector<vector<vector<vector<float>>>> activations_3d; // for conv
    float lr;
    Network(float learning_rate = 0.001f) : lr(learning_rate) {}

    void add_conv(int in_h, int in_w, int in_c, int out_h, int out_w, int out_c, int kernel, int stride, int padding) {
        layers.emplace_back(ConvolutionalLayer(in_h, in_w, in_c, out_h, out_w, out_c, kernel, stride, padding));
        activations_3d.resize(layers.size());
    }
    void add_flatten() {
        layers.emplace_back(); // FLATTEN
    }
    void add_dense(int input_size, int output_size, ActivationType activation = ActivationType::RELU) {
        layers.emplace_back(DenseLayer(input_size, output_size, activation));
        activations_1d.resize(layers.size());
    }
    // Build a simple conv net: conv, flatten, dense, output
    void build_cnn() {
        layers.clear();
        // CIFAR-10: 32x32x3
        add_conv(32, 32, 3, 28, 28, 8, 5, 1, 0); // 8 filters, 5x5, stride 1, no padding
        add_flatten();
        add_dense(28*28*8, 64, ActivationType::RELU);
        add_dense(64, 10, ActivationType::LINEAR);
        activations_3d.resize(layers.size());
        activations_1d.resize(layers.size());
    }
    // Forward pass
    vector<float> forward(const vector<float>& x) {
        vector<vector<vector<float>>> current3d;
        vector<float> current1d = x;
        for (int i = 0; i < layers.size(); ++i) {
            if (layers[i].type == LayerType::CONV) {
                if (i == 0) current3d = reshape1Dto3D(current1d, 3, 32, 32);
                current3d = layers[i].conv.forward(current3d);
                activations_3d[i] = current3d;
            } else if (layers[i].type == LayerType::FLATTEN) {
                current1d = flatten3D(current3d);
                activations_1d[i] = current1d;
            } else if (layers[i].type == LayerType::DENSE) {
                current1d = layers[i].dense.forward(current1d);
                activations_1d[i] = current1d;
            }
        }
        return current1d;
    }
    // Backward pass
    void backward(const vector<float>& grad_output) {
        vector<float> grad1d = grad_output;
        vector<vector<vector<float>>> grad3d;
        for (int i = layers.size() - 1; i >= 0; --i) {
            if (layers[i].type == LayerType::DENSE) {
                grad1d = layers[i].dense.backward(grad1d, lr);
            } else if (layers[i].type == LayerType::FLATTEN) {
                // Unflatten grad1d to grad3d
                int prev = i-1;
                int c = activations_3d[prev].size();
                int h = activations_3d[prev][0].size();
                int w = activations_3d[prev][0][0].size();
                grad3d = reshape1Dto3D(grad1d, c, h, w);
            } else if (layers[i].type == LayerType::CONV) {
                grad3d = layers[i].conv.backward(grad3d, lr);
            }
        }
    }
    float train_sample(const vector<float>& x, int y) {
        auto output = forward(x);
        auto pred = softmax(output);
        float loss = cross_entropy(pred, y);
        auto grad = softmax_loss_backward(pred, y);
        backward(grad);
        return loss;
    }
    int predict(const vector<float>& x) {
        auto output = forward(x);
        auto pred = softmax(output);
        return argmax(pred);
    }
    float evaluate(const vector<vector<float>>& X, const vector<uint8_t>& y) {
        int correct = 0;
        for (int i = 0; i < X.size(); ++i) {
            if (predict(X[i]) == y[i]) correct++;
        }
        return (float)correct / X.size();
    }
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
            if (shuffle_data) shuffle(indices.begin(), indices.end(), g);
            for (int idx = 0; idx < n_train; ++idx) {
                int i = indices[idx];
                total_loss += train_sample(X_train[i], y_train[i]);
            }
            float train_acc = evaluate(X_train, y_train);
            msg("Epoch " + to_string(epoch + 1) + "/" + to_string(epochs)
                 + " - Loss: " + to_string(total_loss / n_train)
                 + " - Train Acc: " + to_string(train_acc * 100) + "%\n");
            if ((epoch + 1) % eval_every == 0 || epoch == epochs - 1) {
                float test_acc = evaluate(X_test, y_test);
                msg("  Test Accuracy: " + to_string(test_acc * 100) + "%\n");
            }
        }
    }
    void set_learning_rate(float new_lr) { lr = new_lr; }
    // Save only dense layers for now
    void save(const string& prefix) {
        int dense_idx = 0;
        for (int i = 0; i < layers.size(); ++i) {
            if (layers[i].type == LayerType::DENSE) {
                string filename = prefix + "_layer_" + to_string(dense_idx) + ".bin";
                save_layer(layers[i].dense, filename);
                dense_idx++;
            }
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
    // Build a simple CNN: Conv(8x5x5)->Flatten->Dense(64)->Dense(10)
    network.build_cnn();
    // Train the network
    network.train(train_images, train_labels, test_images, test_labels, 10, true, 2);
    // Save the trained network
    network.save("cifar_network");
    return 0;
}