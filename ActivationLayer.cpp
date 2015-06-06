#include <algorithm>
#include <cassert>
#include <cmath>
#include "ActivationLayer.hpp"

ActivationLayer::ActivationLayer(int numFM, int mapSize) : Layer(mapSize, mapSize, numFM, numFM) {}


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
    assert(input.at(0).size() == mapSize*mapSize);
    assert(input.size() == numFM);
    
    this->input = &input;
    
    for (int fm = 0; fm < numFM; ++fm)
    {
        std::transform(input.at(fm).begin(), input.at(fm).end(), output.at(fm).begin(), 
                        [this](int x){return this->activationFunction(x);});
    }
    return output;
}

vvf& ActivationLayer::backPropagate(const vvf &error)
{
    assert(error.at(0).size() == mapSize*mapSize);
    assert(error.size() == numFM);

    for (int fm = 0; fm < numFM; ++fm)
    {
        std::transform(input->at(fm).begin(), input->at(fm).end(), prevError.at(fm).begin(),
                        [this](int x){return this->activationFunctionDerivative(x);});
        for (int elem = 0, size = mapSize*mapSize; elem < size; ++elem)
        {
            prevError.at(fm).at(elem) *= error.at(fm).at(elem);
        }
    }
    return output;

}
