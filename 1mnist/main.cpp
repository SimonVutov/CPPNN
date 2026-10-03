#include "../include/data.hpp"
#include "../include/options.hpp"
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
        if (in_size < 0 || out_size < 0 || ((in_size == 0) != (out_size == 0)))
            throw invalid_argument("Invalid dense layer dimensions");
        weights.resize(out_size, vector<float>(in_size));
        biases.resize(out_size);
        pre_activation.resize(out_size);
        auto& gen = training_rng;
        uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &row : weights)
            for (auto &w : row)
                w = dist(gen);
    }

    vector<float> forward(const vector<float>& x, ActivationFunc activation_func = nullptr) {
        if (!in_size || x.size() != static_cast<size_t>(in_size))
            throw invalid_argument("Dense input shape mismatch");
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
        if (input.size() != static_cast<size_t>(in_size) || grad_out.size() != static_cast<size_t>(out_size) || !in_size)
            throw invalid_argument("Dense backward requires a forward pass and matching gradient");
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

void save_layer(const DenseLayer& L, const string& file) {
    ofstream f(file, ios::binary);
    if(!f) throw runtime_error("Cannot write model: " + file);
    int in = L.in_size, out = L.out_size;
    f.write((char*)&in,  sizeof(int));
    f.write((char*)&out, sizeof(int));
    for (auto& row : L.weights)  f.write((char*)row.data(), row.size()*sizeof(float));
    f.write((char*)L.biases.data(), L.biases.size()*sizeof(float));
}

#ifndef CPPNN_NO_MAIN
int main(int argc, char** argv) try {
    auto args=options(argc,argv,"1mnist",5);
    if(args.help) return 0;
    int n_train, n_test, rows, cols;
    auto train_images = read_images((args.data/"train-images-idx3-ubyte").string(), n_train, rows, cols);
    auto train_labels = read_labels((args.data/"train-labels-idx1-ubyte").string(), n_train);
    auto test_images = read_images((args.data/"t10k-images-idx3-ubyte").string(), n_test, rows, cols);
    auto test_labels = read_labels((args.data/"t10k-labels-idx1-ubyte").string(), n_test);

    if(train_images.size()!=train_labels.size() || test_images.size()!=test_labels.size())
        throw runtime_error("Image/label count mismatch");
    if(args.limit) {n_train=min(n_train,args.limit); n_test=min(n_test,args.limit);}
    DenseLayer l1(784, 12);
    DenseLayer l2(12, 10);
    float lr = 0.005f;
    ActivationFunc activation_func = leaky_relu_activation;
    DerivativeFunc derivative_func = leaky_relu_derivative;

    // Open result file for writing
    ofstream result_file(args.output/"result.txt");
    if(!result_file) throw runtime_error("Cannot write training log");
    
    for (int epoch = 0; epoch < args.epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < n_train; i++) {
            auto x = train_images[i];
            int y = train_labels[i];

            // Use ReLU for first layer, no activation for output layer
            auto h = l1.forward(x, activation_func);
            auto out = l2.forward(h); // No activation function (linear output)
            auto pred = softmax(out);
            total_loss += cross_entropy(pred, y);
            correct += (argmax(pred) == y);

            auto grad = softmax_loss_backward(pred, y);
            auto grad_l2 = l2.backward(grad, lr); // No derivative function (linear)
            l1.backward(grad_l2, lr, derivative_func);
        }
        
        string epoch_msg = "Epoch " + to_string(epoch + 1) + " - Loss: " + to_string(total_loss / n_train) + 
                          ", Accuracy: " + to_string((float)correct / n_train * 100) + "%\n";
        
        cout << epoch_msg;
        result_file << epoch_msg;
    }
    
    // Evaluate on test set
    int correct = 0;
    for (int i = 0; i < n_test; ++i) {
        auto h = l1.forward(test_images[i], activation_func);
        auto out = l2.forward(h); // No activation
        auto pred = softmax(out);
        if (argmax(pred) == test_labels[i])
        ++correct;
    }
    
    string test_msg = "Test Accuracy: " + to_string((float)correct / n_test * 100) + "%\n";
    cout << test_msg;
    result_file << test_msg;
    
    result_file.close();
    
    save_layer(l1, (args.output/"l1.bin").string());
    save_layer(l2, (args.output/"l2.bin").string());
    return 0;
}
catch(const exception& error) {cerr<<"error: "<<error.what()<<"\n"; return 1;}
#endif
