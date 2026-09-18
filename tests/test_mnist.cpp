#define CPPNN_NO_MAIN
#include "../1mnist/main.cpp"
#include "check.hpp"
int main() {
    DenseLayer layer(2,2);
    layer.weights={{1,2},{-1,3}};
    layer.biases={0.5f,-0.5f};
    auto value=layer.forward({2,1},leaky_relu_activation);
    CHECK(value[0]==4.5f && value[1]==0.5f);
    auto gradient=layer.backward({1,2},0.1f,leaky_relu_derivative);
    CHECK(gradient[0]==-1 && gradient[1]==8);
    CHECK(std::abs(layer.weights[0][0]-0.8f)<1e-6);
    auto probabilities=softmax({1000,1001});
    CHECK(std::abs(probabilities[0]+probabilities[1]-1)<1e-6);
    CHECK(std::isfinite(cross_entropy(probabilities,0)));
}
