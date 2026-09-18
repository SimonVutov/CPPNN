#include "../include/data.hpp"
#include "../include/options.hpp"
#include <chrono>
#include <ctime>
#include <numeric>
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

std::filesystem::path data_directory="2CIFAR10/cifar-10-batches-bin", output_directory="runs/cifar";

void msg(const string& message) {
    // add time and date to the message
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    string time_str = ctime(&now_time);
    time_str.pop_back(); // remove newline
    const string newMessage = time_str + " : " + message;

    // open in append mode
    ofstream result_file(output_directory/"result.txt", ios::app);
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

pair<vector<vector<float>>, vector<uint8_t>> load_cifar_train() {
    vector<vector<float>> all_images;
    vector<uint8_t> all_labels;
    
    // Load all 5 training batches
    for (int batch = 1; batch <= 5; ++batch) {
        string filename = (data_directory/("data_batch_" + to_string(batch) + ".bin")).string();
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
    string filename = (data_directory/"test_batch.bin").string();
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
        if (in_size < 0 || out_size < 0 || ((in_size == 0) != (out_size == 0)))
            throw invalid_argument("Invalid dense layer dimensions");
        weights.resize(out_size, vector<float>(in_size));
        biases.resize(out_size);
        pre_activation.resize(out_size);
        output.resize(out_size);
        
        // Set activation functions
        auto [act_func, deriv_func] = get_activation_functions(activation_type);
        activation_func = act_func;
        derivative_func = deriv_func;
        
        // Xavier/Glorot initialization
        auto& gen = training_rng;
        float limit = in_size+out_size ? sqrt(6.0f / (in_size + out_size)) : 0.0f;
        uniform_real_distribution<float> dist(-limit, limit);
        
        for (auto &row : weights)
            for (auto &w : row)
                w = dist(gen);
        
        // Initialize biases to zero
        fill(biases.begin(), biases.end(), 0.0f);
    }

    vector<float> forward(const vector<float>& x) {
        if (!in_size || x.size() != static_cast<size_t>(in_size))
            throw invalid_argument("Dense input shape mismatch");
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
        if (input.size() != static_cast<size_t>(in_size) || grad_out.size() != static_cast<size_t>(out_size) || !in_size)
            throw invalid_argument("Dense backward requires a forward pass and matching gradient");
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
        const bool placeholder = in_h==0 && in_w==0 && in_c==0 && out_h==0 && out_w==0 && out_c==0 && kernel_size==0 && stride==0 && padding==0;
        if (!placeholder && (in_h<=0 || in_w<=0 || in_c<=0 || out_c<=0 || out_h<=0 || out_w<=0 || kernel_size<=0 || stride<=0 || padding<0 ||
            int64_t(in_h)+2LL*padding<kernel_size || int64_t(in_w)+2LL*padding<kernel_size ||
            int64_t(in_h)+2LL*padding>INT32_MAX || int64_t(in_w)+2LL*padding>INT32_MAX ||
            out_h!=(int64_t(in_h)+2LL*padding-kernel_size)/stride+1 || out_w!=(int64_t(in_w)+2LL*padding-kernel_size)/stride+1))
            throw invalid_argument("Invalid convolution dimensions");
        weights.resize(out_c, vector<vector<vector<float>>>(in_c, vector<vector<float>>(kernel_size, vector<float>(kernel_size))));
        biases.resize(out_c);
        // Xavier/Glorot initialization
        auto& gen = training_rng;
        float fan_in = in_c * kernel_size * kernel_size;
        float fan_out = out_c;
        float limit = fan_in+fan_out ? sqrt(6.0f / (fan_in + fan_out)) : 0.0f;
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
        if (!in_c || x.size()!=static_cast<size_t>(in_c)) throw invalid_argument("Convolution channel mismatch");
        for (const auto& channel:x) {
            if(channel.size()!=static_cast<size_t>(in_h)) throw invalid_argument("Convolution height mismatch");
            for(const auto& row:channel) if(row.size()!=static_cast<size_t>(in_w)) throw invalid_argument("Convolution width mismatch");
        }
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
        if(grad_out.size()!=static_cast<size_t>(out_c)) throw invalid_argument("Convolution gradient channel mismatch");
        for(const auto& channel:grad_out) {
            if(channel.size()!=static_cast<size_t>(out_h)) throw invalid_argument("Convolution gradient height mismatch");
            for(const auto& row:channel) if(row.size()!=static_cast<size_t>(out_w)) throw invalid_argument("Convolution gradient width mismatch");
        }
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
    if(c<=0 || h<=0 || w<=0 || (uint64_t(c)*h > x.size()/static_cast<size_t>(w) || uint64_t(c)*h*w != x.size())) throw invalid_argument("Invalid reshape dimensions");
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
  if (logits.empty()) throw invalid_argument("Softmax requires nonempty logits");
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
  if (label < 0 || static_cast<size_t>(label) >= pred.size()) throw invalid_argument("Invalid class label");
  return -log(pred[label] + 1e-8f);
}

vector<float> softmax_loss_backward(const vector<float> &pred, int label) {
  if (label < 0 || static_cast<size_t>(label) >= pred.size()) throw invalid_argument("Invalid class label");
  vector<float> grad = pred;
  grad[label] -= 1.0f;
  return grad;
}

int argmax(const vector<float> &v) {
  if (v.empty()) throw invalid_argument("argmax requires nonempty input");
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

// Add LayerKind and LayerConfig for user-specified network
enum class LayerKind { CONV, FLATTEN, DENSE };
struct LayerConfig {
    LayerKind kind;
    int in_h = 0, in_w = 0, in_c = 0, out_h = 0, out_w = 0, out_c = 0, kernel = 0, stride = 0, padding = 0;
    int in_size = 0, out_size = 0;
    ActivationType activation = ActivationType::LINEAR;
    LayerConfig(LayerKind k) : kind(k) {}
    // Static helpers for user-friendly construction
    static LayerConfig Conv(int in_h, int in_w, int in_c, int out_h, int out_w, int out_c, int kernel, int stride, int padding) {
        LayerConfig c(LayerKind::CONV);
        c.in_h = in_h; c.in_w = in_w; c.in_c = in_c;
        c.out_h = out_h; c.out_w = out_w; c.out_c = out_c;
        c.kernel = kernel; c.stride = stride; c.padding = padding;
        return c;
    }
    static LayerConfig Flatten() {
        return LayerConfig(LayerKind::FLATTEN);
    }
    static LayerConfig Dense(int in_size, int out_size, ActivationType activation) {
        LayerConfig d(LayerKind::DENSE);
        d.in_size = in_size; d.out_size = out_size; d.activation = activation;
        return d;
    }
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
    void build_from_config(const vector<LayerConfig>& configs) {
        layers.clear();
        for (const auto& cfg : configs) {
            if (cfg.kind == LayerKind::CONV) {
                add_conv(cfg.in_h, cfg.in_w, cfg.in_c, cfg.out_h, cfg.out_w, cfg.out_c, cfg.kernel, cfg.stride, cfg.padding);
            } else if (cfg.kind == LayerKind::FLATTEN) {
                add_flatten();
            } else if (cfg.kind == LayerKind::DENSE) {
                add_dense(cfg.in_size, cfg.out_size, cfg.activation);
            }
        }
        activations_3d.resize(layers.size());
        activations_1d.resize(layers.size());
    }
    // Forward pass
    vector<float> forward(const vector<float>& x, bool verbose = false) {
        vector<vector<vector<float>>> current3d;
        vector<float> current1d = x;
        for (int i = 0; i < layers.size(); ++i) {
            string layer_type;
            switch (layers[i].type) {
                case LayerType::CONV: layer_type = "Conv"; break;
                case LayerType::FLATTEN: layer_type = "Flatten"; break;
                case LayerType::DENSE: layer_type = "Dense"; break;
            }
            if (verbose) cout << "[Forward] Entering Layer " << i << ": " << layer_type << endl;
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
    float train_sample(const vector<float>& x, int y, bool verbose = false) {
        auto output = forward(x, verbose);
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
        auto& g = training_rng;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            if (shuffle_data) shuffle(indices.begin(), indices.end(), g);
            for (int idx = 0; idx < n_train; ++idx) {
                int i = indices[idx];
                total_loss += train_sample(X_train[i], y_train[i], (idx + 1) % 5000 == 0);
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
    void save_dense_weights(const string& prefix) {
        int dense_idx = 0;
        for (int i = 0; i < layers.size(); ++i) {
            if (layers[i].type == LayerType::DENSE) {
                string filename = prefix + "_layer_" + to_string(dense_idx) + ".bin";
                save_layer(layers[i].dense, filename);
                dense_idx++;
            }
        }
        msg("Dense weights saved (convolutions excluded), prefix: " + prefix + "\n");
    }
private:
    void save_layer(const DenseLayer& L, const string& file) {
        ofstream f(file, ios::binary);
        if(!f) throw runtime_error("Cannot write model: " + file);
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

#ifndef CPPNN_NO_MAIN
int main(int argc, char** argv) try {
    auto args=options(argc,argv,"2CIFAR10/cifar-10-batches-bin",10);
    if(args.help) return 0;
    data_directory=args.data; output_directory=args.output;
    // Load CIFAR-10 data
    auto [train_images, train_labels] = load_cifar_train();
    auto [test_images, test_labels] = load_cifar_test();
    if(args.limit) {
        train_images.resize(min(train_images.size(),size_t(args.limit))); train_labels.resize(train_images.size());
        test_images.resize(min(test_images.size(),size_t(args.limit))); test_labels.resize(test_images.size());
    }
    msg("Training samples: " + to_string(train_images.size()) + "\n" + "Test samples: " + to_string(test_images.size()) + "\n" + "Input size: " + to_string(train_images[0].size()) + "\n");
    // User specifies the network architecture here:
    vector<LayerConfig> user_layers = {
        LayerConfig::Conv(32, 32, 3, 28, 28, 8, 5, 1, 0),
        LayerConfig::Conv(28, 28, 8, 24, 24, 16, 5, 1, 0),
        LayerConfig::Conv(24, 24, 16, 20, 20, 32, 5, 1, 0),
        LayerConfig::Flatten(),
        LayerConfig::Dense(20*20*32, 512, ActivationType::LEAKY_RELU),
        LayerConfig::Dense(512, 128, ActivationType::LEAKY_RELU),
        LayerConfig::Dense(128, 10, ActivationType::LINEAR)
    };
    Network network(0.001f); // learning rate
    network.build_from_config(user_layers);
    network.train(train_images, train_labels, test_images, test_labels, args.epochs, true, 2);
    network.save_dense_weights((args.output/"cifar_network").string());
    return 0;
}
catch(const exception& error) {cerr<<"error: "<<error.what()<<"\n"; return 1;}
#endif
