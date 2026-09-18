#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

// g++ -std=c++17 -O2 digit_ui.cpp -o digit_ui \$(pkg-config --cflags --libs opencv4)
// ./digit_ui

/* ---------- network skeleton (forward-only) ---------- */
struct DenseLayer {
    int in_size, out_size;
    vector<vector<float>> W;
    vector<float> b, out;

    void load(const string& file) {
        ifstream f(file, ios::binary);
        f.read((char*)&in_size,  sizeof(int));
        f.read((char*)&out_size, sizeof(int));
        if(!f || in_size<=0 || out_size<=0 || in_size>784 || out_size>784)
            throw runtime_error("Invalid model header: " + file);
        W.assign(out_size, vector<float>(in_size));
        b.resize(out_size);
        for (auto& row : W) f.read((char*)row.data(), row.size()*sizeof(float));
        f.read((char*)b.data(), b.size()*sizeof(float));
        if(!f) throw runtime_error("Truncated model: " + file);
    }
    vector<float> forward(const vector<float>& x, bool relu=true) {
        if(x.size()!=static_cast<size_t>(in_size)) throw runtime_error("Input shape mismatch");
        out.assign(out_size, 0.0f);
        for (int i=0;i<out_size;++i) {
            float s = b[i];
            for (int j=0;j<in_size;++j) s += W[i][j]*x[j];
            out[i] = relu ? max(0.01f*s,s) : s;
        }
        return out;
    }
};

vector<float> softmax(const vector<float>& z) {
    float m = *max_element(z.begin(), z.end());
    float sum = 0;  vector<float> p(z.size());
    for (int i=0;i<z.size();++i){ p[i] = exp(z[i]-m); sum += p[i]; }
    for (float& v:p) v/=sum;
    return p;
}
int argmax(const vector<float>& v){
    return max_element(v.begin(), v.end()) - v.begin();
}

/* -------------- simple paint canvas ------------- */
const int CANVAS = 280;          // 10× MNIST resolution
Mat canvas(CANVAS, CANVAS, CV_8UC1, Scalar(0));
bool drawing = false;

void mouse(int event,int x,int y,int,void*) {
    if (event == EVENT_LBUTTONDOWN) drawing = true;
    else if (event == EVENT_LBUTTONUP) drawing = false;
    if (drawing)
        circle(canvas, Point(x,y), 12, Scalar(255), -1, LINE_AA); // thick brush
}

/* -------------- main ------------- */
int main(int argc, char** argv) try {
    const filesystem::path model_dir = argc>1 ? argv[1] : "runs/mnist";
    /* load model */
    DenseLayer l1, l2;
    l1.load((model_dir/"l1.bin").string());
    l2.load((model_dir/"l2.bin").string());

    namedWindow("Draw digit (press P to predict, C to clear, Esc to quit)");
    setMouseCallback("Draw digit (press P to predict, C to clear, Esc to quit)", mouse);

    while (true) {
        imshow("Draw digit (press P to predict, C to clear, Esc to quit)", canvas);
        char k = (char)waitKey(15);

        if (k == 27) break;                      // ESC
        if (k == 'c' || k == 'C') canvas.setTo(0);

        if (k == 'p' || k == 'P') {
            /* down-sample 280×280 → 28×28, flatten & normalise */
            Mat small;
            resize(canvas, small, Size(28,28), 0,0, INTER_AREA);
            vector<float> x(784);
            for (int r=0;r<28;++r)
                for (int c=0;c<28;++c)
                    x[r*28+c] = small.at<uchar>(r,c)/255.0f;

            /* forward */
            auto h   = l1.forward(x);            // ReLU
            auto out = l2.forward(h,false);      // raw logits
            auto prob = softmax(out);
            int pred  = argmax(prob);

            cout << "Predicted digit: " << pred << "  (";
            for(int i=0;i<10;++i) cout<<fixed<<setprecision(2)<<prob[i]<<" ";
            cout << ")\n";
        }
    }
    destroyAllWindows();
    return 0;
}

catch(const exception& error) {cerr<<"error: "<<error.what()<<"\n";return 1;}
