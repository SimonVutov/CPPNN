#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline int32_t read_int(std::ifstream& file) {
    unsigned char bytes[4]{};
    if (!file.read(reinterpret_cast<char*>(bytes), 4)) throw std::runtime_error("Truncated IDX header");
    const uint32_t value = (uint32_t(bytes[0])<<24) | (uint32_t(bytes[1])<<16) | (uint32_t(bytes[2])<<8) | bytes[3];
    if (value > INT32_MAX) throw std::runtime_error("Invalid IDX dimension");
    return static_cast<int32_t>(value);
}
inline std::vector<std::vector<float>> read_images(const std::string& path, int& count, int& rows, int& cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open " + path);
    if (read_int(file) != 2051) throw std::runtime_error("Invalid IDX image magic");
    count=read_int(file); rows=read_int(file); cols=read_int(file);
    if (count<=0 || rows!=28 || cols!=28 || std::filesystem::file_size(path)!=16+uint64_t(count)*784)
        throw std::runtime_error("Invalid MNIST image dimensions or file size");
    std::vector<std::vector<float>> images(count, std::vector<float>(784));
    for (auto& image:images) for(auto& value:image) {
        unsigned char pixel=0;
        if (!file.read(reinterpret_cast<char*>(&pixel),1)) throw std::runtime_error("Truncated image");
        value=pixel/255.0f;
    }
    return images;
}
inline std::vector<uint8_t> read_labels(const std::string& path, int& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open " + path);
    if(read_int(file)!=2049) throw std::runtime_error("Invalid IDX label magic");
    count=read_int(file);
    if(count<=0 || std::filesystem::file_size(path)!=8+uint64_t(count)) throw std::runtime_error("Invalid label file size");
    std::vector<uint8_t> labels(count);
    if(!file.read(reinterpret_cast<char*>(labels.data()),count)) throw std::runtime_error("Truncated labels");
    for(auto label:labels) if(label>=10) throw std::runtime_error("Invalid class label");
    return labels;
}
struct CifarData {
    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;
};
inline CifarData read_cifar_batch(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if(!file) throw std::runtime_error("Cannot open " + path);
    auto bytes=std::filesystem::file_size(path);
    if(!bytes || bytes%3073) throw std::runtime_error("Invalid CIFAR batch size");
    CifarData data;
    const auto count=bytes/3073;
    data.images.resize(count, std::vector<float>(3072)); data.labels.resize(count);
    for(size_t i=0;i<count;++i) {
        if(!file.read(reinterpret_cast<char*>(&data.labels[i]),1) || data.labels[i]>=10)
            throw std::runtime_error("Invalid CIFAR label");
        unsigned char image[3072];
        if(!file.read(reinterpret_cast<char*>(image),3072)) throw std::runtime_error("Truncated CIFAR image");
        for(size_t j=0;j<3072;++j) data.images[i][j]=image[j]/255.0f;
    }
    return data;
}
