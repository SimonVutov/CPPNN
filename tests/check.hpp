#pragma once
#include <stdexcept>
#include <cmath>
#define CHECK(...) do { if(!(__VA_ARGS__)) throw std::runtime_error("Check failed: " #__VA_ARGS__); } while(false)
template<class F> void fails(F f) { bool caught=false; try {f();} catch(const std::exception&) {caught=true;} CHECK(caught); }
