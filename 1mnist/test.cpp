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

// g++ -O2 -std=c++17 test.cpp -o test && ./test

using namespace std;

int main() {
    int n_train, n_test, rows, cols;
    auto train_images = read_images("train-images-idx3-ubyte", n_train, rows, cols);
    auto train_labels = read_labels("train-labels-idx1-ubyte", n_train);
    auto test_images = read_images("t10k-images-idx3-ubyte", n_test, rows, cols);
    auto test_labels = read_labels("t10k-labels-idx1-ubyte", n_test);

    cout << "Number of training images: " << n_train << endl;
    cout << "Number of test images: " << n_test << endl;
    cout << "Rows: " << rows << endl;
    cout << "Cols: " << cols << endl;

    // print first 5 labels
    for (int i = 0; i < 5; ++i) {
        cout << "Label " << i << ": " << (int)train_labels[i] << endl;
    }

    // print first 5 images
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            cout << (train_images[i][j] > 0.1 ? 1 : 0) << " ";
            if (j % cols == cols - 1) {
                cout << endl;
            }
        }
        cout << endl;
    }

    float learning_rate = 0.05f;

    vector<vector<float>> l1 (16, vector<float>(784));
    vector<vector<float>> l2 (10, vector<float>(16));
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    for (auto& row : l1)
        for (auto& val : row)
            val = distribution(generator);
    for (auto& row : l2)
        for (auto& val : row)
            val = distribution(generator);

    for (int epoch = 0; epoch < 5; epoch++) {
        float avg_loss = 0;
        for (int image = 0; image < 10000; image++) {
            //784 long string input (the number)
            
            vector<float> l1_output (16, 0);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 784; j++) {
                    l1_output[i] += train_images[image][j] * l1[i][j];
                }
                l1_output[i] = max(l1_output[i], 0.0f); // relu
            }
            
            vector<float> l2_output (10, 0);
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 16; j++) {
                    l2_output[i] += l1_output[j] * l2[i][j];
                }
            }

            // do a softmax
            vector<float> softmax_output(10, 0);
            float sum = 0.0f;
            for (int i = 0; i < 10; i++) {
                softmax_output[i] = exp(l2_output[i]);
                sum += softmax_output[i];
            }
            for (float& f : softmax_output) f /= sum;

            float loss = -log(softmax_output[train_labels[image]] + 1e-8f);
            avg_loss += loss;

            vector<float> grad_output = softmax_output;
            grad_output[train_labels[image]] -= 1.0f; // Subtract 1 from the correct class

            vector<vector<float>> l2_gradients(10, vector<float>(16));
            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 16; ++j) {
                    l2_gradients[i][j] = grad_output[i] * l1_output[j];
                }
            }

            vector<float> l1_error(16, 0.0f);
            for (int j = 0; j < 16; ++j) {
                for (int i = 0; i < 10; ++i) {
                    l1_error[j] += grad_output[i] * l2[i][j];
                }
                // Apply derivative of ReLU activation
                l1_error[j] *= (l1_output[j] > 0) ? 1.0f : 0.0f;
            }
            vector<vector<float>> l1_gradients(16, vector<float>(784));
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 784; ++j) {
                    l1_gradients[i][j] = l1_error[i] * train_images[image][j];
                }
            }
            // Update l2 weights
            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 16; ++j) {
                    l2[i][j] -= learning_rate * l2_gradients[i][j];
                }
            }

            // Update l1 weights
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 784; ++j) {
                    l1[i][j] -= learning_rate * l1_gradients[i][j];
                }
            }
        }
        cout << avg_loss / 10000.0f << endl;
    }

    cout << "Predictions for first 5 images: " << endl;
    for (int image = 0; image < 5; ++image) {
        vector<float> l1_output(16, 0);
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 784; j++) {
                l1_output[i] += train_images[image][j] * l1[i][j];
            }
            l1_output[i] = max(l1_output[i], 0.0f); // ReLU
        }
        vector<float> l2_output(10, 0);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 16; j++) {
                l2_output[i] += l1_output[j] * l2[i][j];
            }
        }
        vector<float> softmax_output(10, 0);
        float sum = 0.0f;
        for (int i = 0; i < 10; i++) {
            softmax_output[i] = exp(l2_output[i]);
            sum += softmax_output[i];
        }
        for (float& f : softmax_output) f /= sum;
        for (float j : softmax_output) cout << fixed << setprecision(2) << j << " ";
        cout << endl;
    }

    return 0;
}