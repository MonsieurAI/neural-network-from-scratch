#pragma once
#include <vector>

class Tensor {
    private:
        std::vector<float> _data;
        std::vector<size_t> _shape;
        std::vector<size_t> _stride;
    public:
        Tensor(float data);
        Tensor(std::vector<float> data);
        Tensor(std::vector<std::vector<float>> data);
};