#include "../include/tensor.h"
#include <stdexcept>

Tensor::Tensor(float data) : _data{data}, _shape{}, _stride{} {};

Tensor::Tensor(std::vector<float> data) : _data(data), _shape{data.size()}, _stride{1} {};

Tensor::Tensor(std::vector<std::vector<float>> data)
    : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1} {

        // Check dimension consistency
        size_t expected_size = data[0].size();
        for (size_t i = 0; i < data.size(); i++) {
            if (data[i].size() != expected_size) {
                throw std::invalid_argument("Dimensions are incositent");
            }
        }

        // Store data in row major format
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[0].size(); j++) {
                _data.push_back(data[i][j]);
            }
        }
    }

const float &Tensor::item() const {
    if (_shape.size() == 0) {
        return _data[0];
    }
    else {
        throw std::runtime_error("Tensor must be 0D for .item()");
    }
}

float &Tensor::item() {
    if (_shape.size() == 0) {
        return _data[0];
    }
    else {
        throw std::runtime_error("Tensor must be 0D for .item()");
    }
}