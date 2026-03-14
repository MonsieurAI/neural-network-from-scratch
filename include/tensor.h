#pragma once
#include <vector>
#include <ostream>
#include <memory>

class Tensor {
    private:
        std::vector<float> _data;
        std::vector<size_t> _shape;
        std::vector<size_t> _stride;
    public:
        Tensor(float data);
        Tensor(std::vector<float> data);
        Tensor(std::vector<std::vector<float>> data);
        const float &item() const;
        float &item();
        const std::vector<size_t> &shape() const;
        const float &operator()(size_t i) const;
        float &operator()(size_t i);
        const float &operator()(size_t i, size_t j) const;
        float &operator()(size_t i, size_t j);
        friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
        std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
};