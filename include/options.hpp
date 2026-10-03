#pragma once
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
inline std::mt19937 training_rng(42);
struct Options {
    std::filesystem::path data, output;
    int epochs=5, limit=0;
    bool help=false;
};
inline Options options(int argc, char** argv, const char* default_data, int epochs) {
    Options result{default_data, "runs/cpp", epochs, 0, false};
    unsigned seed=42;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {result.help=true; break;}
        if(i+1==argc) throw std::invalid_argument("Missing option value");
        std::string value=argv[++i];
        if(key=="--data") result.data=value;
        else if(key=="--output") result.output=value;
        else {
            size_t used=0;
            int number=std::stoi(value,&used);
            if(used!=value.size() || number<0) throw std::invalid_argument("Invalid numeric option");
            if(key=="--epochs" && number>0) result.epochs=number;
            else if(key=="--limit" && number>0) result.limit=number;
            else if(key=="--seed") seed=static_cast<unsigned>(number);
            else throw std::invalid_argument("Unknown or invalid option: " + key);
        }
    }
    if(result.help) std::cout<<"--data PATH --output PATH --epochs N --limit N --seed N\n";
    else std::filesystem::create_directories(result.output);
    training_rng.seed(seed);
    return result;
}
