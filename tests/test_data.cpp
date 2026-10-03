#include "../include/data.hpp"
#include "check.hpp"
int main() {
    const std::string path="test_batch_fixture.bin";
    {std::ofstream file(path,std::ios::binary); file.put(3); for(int i=0;i<3072;++i) file.put(char(255));}
    auto batch=read_cifar_batch(path);
    CHECK(batch.labels.size()==1 && batch.labels[0]==3 && batch.images[0][0]==1);
    {std::ofstream file(path,std::ios::binary); file<<"bad";}
    fails([&]{read_cifar_batch(path);});
    int n,r,c;
    fails([&]{read_images(path,n,r,c);});
    fails([&]{read_labels(path,n);});
    std::filesystem::remove(path);
    fails([&]{read_cifar_batch(path);});
    const std::string image="test_images_fixture.idx", label="test_labels_fixture.idx";
    auto integer=[](std::ofstream& f,int v){for(int s=24;s>=0;s-=8) f.put(char((v>>s)&255));};
    {std::ofstream f(image,std::ios::binary); integer(f,2051);integer(f,1);integer(f,28);integer(f,28);for(int i=0;i<784;++i) f.put(char(255));}
    {std::ofstream f(label,std::ios::binary);integer(f,2049);integer(f,1);f.put(9);}
    CHECK(read_images(image,n,r,c)[0][0]==1 && n==1 && r==28 && c==28);
    CHECK(read_labels(label,n)[0]==9);
    std::filesystem::remove(image);std::filesystem::remove(label);
}
