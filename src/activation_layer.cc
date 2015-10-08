#include "activation_layer.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace cnn {

ActivationLayer::ActivationLayer(int num_feature_maps, int map_size) 
    : Layer(map_size, map_size, num_feature_maps, num_feature_maps) {}

float TanhLayer::activationFunction(float x)
{
    return 1.7159 * std::tanh(2.0/3 * x);
}

float TanhLayer::activationFunctionDerivative(float x)
{
    return 1.444 * (1 - std::pow(std::tanh(2.0/3 * x), 2));
}

vvf& ActivationLayer::forwardPass(const vvf &input)
{
    assert(input.at(0).size() == map_size_*map_size_);
    assert(input.size() == num_feature_maps_);
    
    this->input_ = &input;
    
    for (int feature_map = 0; feature_map < num_feature_maps_; ++feature_map)
    {
        std::transform(input.at(feature_map).begin(), input.at(feature_map).end(), output_.at(feature_map).begin(), 
                       [this](int x){ return this->activationFunction(x); });
    }
    return output_;
}

vvf& ActivationLayer::backPropagate(const vvf &error)
{
    assert(error.at(0).size() == map_size_*map_size_);
    assert(error.size() == num_feature_maps_);

    for (int feature_map = 0; feature_map < num_feature_maps_; ++feature_map)
    {
        std::transform(input_->at(feature_map).begin(), input_->at(feature_map).end(), prev_error_.at(feature_map).begin(),
                       [this](int x){ return this->activationFunctionDerivative(x); });
        for (int elem = 0, size = map_size_*map_size_; elem < size; ++elem)
        {
            prev_error_.at(feature_map).at(elem) *= error.at(feature_map).at(elem);
        }
    }
    return output_;
}

} // namespace cnn

