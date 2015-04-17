#include <cmath>
#include <cassert>
#include <random>
#include "Util.hpp"

void ReLUInitializer::init(vd &weights) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(0,2.0/n);
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = distrib(gen);
    }
}


void TestInitializer::init(vd &weights) const
{
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = 1;
    }
}

vvd& SquareCost::calculate(const vvd &output, const vd& expectedOutput)
{
    assert(output.size() == numOutputs);
    assert(expectedOutput.size() == numOutputs);
    
    error = 0;
    for (int i = 0; i < numOutputs; ++i)
    {
        assert(output.at(i).size() == 1);
        prevError.at(i).at(0) = output.at(i).at(0) - expectedOutput.at(i);
        error += 0.5 * std::pow(output.at(i).at(0) - expectedOutput.at(i), 2);
    }
    return prevError;
}
