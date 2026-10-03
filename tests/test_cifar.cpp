#define CPPNN_NO_MAIN
#include "../2CIFAR10/main.cpp"
#include "check.hpp"
int main() {
    DenseLayer dense(2,2,ActivationType::LINEAR);
    dense.weights={{1,2},{-1,3}}; dense.biases={0,0};
    CHECK(dense.forward({2,1})[0]==4);
    auto grad=dense.backward({1,2},0);
    CHECK(grad[0]==-1 && grad[1]==8);
    ConvolutionalLayer conv(3,3,1,2,2,1,2,1,0);
    conv.weights={{{{1,2},{3,4}}}}; conv.biases={0};
    vector<vector<vector<float>>> input={{{1,2,3},{4,5,6},{7,8,9}}};
    auto output=conv.forward(input);
    CHECK(output[0][0][0]==37 && output[0][1][1]==77);
    vector<vector<vector<float>>> delta(1,vector<vector<float>>(2,vector<float>(2,1)));
    auto input_grad=conv.backward(delta,0);
    CHECK(input_grad[0][1][1]==10);
    const float before=conv.weights[0][0][0][0];
    conv.forward(input);conv.backward(delta,0.01f);
    CHECK(std::abs((before-conv.weights[0][0][0][0])/0.01f-12)<1e-4);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        auto probe=conv;
        probe.weights={{{{1,2},{3,4}}}};
        auto changed=input; changed[0][i][j]+=0.01f;
        auto plus=probe.forward(changed);
        changed[0][i][j]-=0.02f;
        auto minus=probe.forward(changed);
        float derivative=0;
        for(int r=0;r<2;++r) for(int c=0;c<2;++c) derivative+=(plus[0][r][c]-minus[0][r][c])/0.02f;
        CHECK(std::abs(derivative-input_grad[0][i][j])<0.002f);
    }
    CHECK(reshape1Dto3D(flatten3D(input),1,3,3)==input);
    fails([]{DenseLayer invalid(-1,2);});
    DenseLayer unchecked(2,2);
    fails([&]{unchecked.forward({1});});
    fails([&]{unchecked.backward({1,2},0);});
    fails([]{softmax({});});
    fails([]{cross_entropy({0.5f,0.5f},3);});
    fails([]{ConvolutionalLayer invalid(3,3,1,5,5,1,2,1,0);});
    fails([&]{conv.forward({});});
    fails([&]{conv.backward({},0);});
    fails([]{reshape1Dto3D({1},2,2,2);});
}
