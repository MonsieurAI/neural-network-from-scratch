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

const std::vector<size_t> &Tensor::shape() const {
    return _shape;
}

const float &Tensor::operator()(size_t i) const {
    if (_shape.size() == 0) {
        throw std::invalid_argument("Cannot index into a scalar. Use .item() instead");
    }
    if (_shape.size() == 1) {
        if (_shape[0] <= i) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for tensor with size " + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("Use two indices for 2D tensors");
}

float &Tensor::operator()(size_t i) {
    if (_shape.size() == 0) {
        throw std::invalid_argument("Cannot index into a scalar. Use .item() instead");
    }
    if (_shape.size() == 1) {
        if (_shape[0] <= i) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for tensor with size " + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("Use two indices for 2D tensors");
}

const float &Tensor::operator()(size_t i, size_t j) const {
    if (_shape.size() == 2) {
        if (_shape[0] <= i) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for tensor with " + std::to_string(_shape[0]) + " rows");
        }
        if (_shape[1] <= j) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for tensor with " + std::to_string(_shape[0]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Use 2 indices for 2D tensors");
}

float &Tensor::operator()(size_t i, size_t j) {
    if (_shape.size() == 2) {
        if (_shape[0] <= i) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for tensor with " + std::to_string(_shape[0]) + " rows");
        }
        if (_shape[1] <= j) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for tensor with " + std::to_string(_shape[0]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Use 2 indeces only for 2D tensors");
}