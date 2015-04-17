#include <algorithm>
#include <cassert>
#include <cmath>
#include "ActivationLayer.hpp"

ActivationLayer::ActivationLayer(int numFM, int mapSize) : numFM(numFM), mapSize(mapSize),
              output(vvd(numFM, vd(mapSize*mapSize))),
              prevError(vvd(numFM, vd(mapSize*mapSize, 0))) {}


float SigmoidLayer::sigmoid(float x)
{
    return 1.7159 * std::tanh(2.0/3 * x);
}

float SigmoidLayer::sigmoidDerivative(float x)
{
    return 1.444 * (1 - std::pow(std::tanh(2.0/3 * x), 2));
}

vvd& SigmoidLayer::forwardPass(const vvd &input)
{
    assert(input.at(0).size() == mapSize*mapSize);
    assert(input.size() == numFM);
    
    this->input = &input;
    
    for (int fm = 0; fm < numFM; ++fm)
    {
        std::transform(input.at(fm).begin(), input.at(fm).end(), output.at(fm).begin(), SigmoidLayer::sigmoid);
    }
    return output;
}

vvd& SigmoidLayer::backPropagate(const vvd &error)
{
    assert(error.at(0).size() == mapSize*mapSize);
    assert(error.size() == numFM);

    for (int fm = 0; fm < numFM; ++fm)
    {
        std::transform(input->at(fm).begin(), input->at(fm).end(), prevError.at(fm).begin(), sigmoidDerivative);
        for (int elem = 0, size = mapSize*mapSize; elem < size; ++elem)
        {
            prevError.at(fm).at(elem) *= error.at(fm).at(elem);
        }
    }
    return output;

}
