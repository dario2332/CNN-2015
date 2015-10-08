#include "convolution_neural_network.h"

namespace cnn {

ConvolutionNeuralNetwork::ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, 
                                                    CostFunction &cost_function, 
                                                    InputManager &input_manager) 
    : layers_(layers), cost_function_(cost_function), input_manager_(input_manager) {}

void ConvolutionNeuralNetwork::feedForward(const vvf &input)
{
    const vvf *current_input = &input;
    for (int layer = 0; layer < layers_.size(); ++layer)
    {
        layers_.at(layer)->forwardPass(*current_input);
        current_input = &(layers_.at(layer)->getOutput());
    }
}

void ConvolutionNeuralNetwork::backPropagate(const vvf &error)
{
    const vvf *current_error = &error;
    for (int layer = layers_.size() - 1; layer >= 0; --layer)
    {
        layers_.at(layer)->backPropagate(*current_error);
        current_error = &layers_.at(layer)->getPrevError();
    }
}

void ConvolutionNeuralNetwork::train(int num_epochs)
{
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        for (int i = 0, n = input_manager_.getInputNum(); i < n; ++i)
        {
            feedForward(input_manager_.getInput(i));
            backPropagate(cost_function_.calculate(layers_.at(layers_.size()-1)->getOutput(), 
                                                   input_manager_.getExpectedOutput(i)));
        }
        notifySupervisors(epoch);
        input_manager_.reset();
    }
}

float ConvolutionNeuralNetwork::getCost(const vf &expected_output) const
{
    cost_function_.calculate(layers_.at(layers_.size()-1)->getOutput(), expected_output);
    return cost_function_.getError();
}

void ConvolutionNeuralNetwork::notifySupervisors(int epoch) const
{
    for (int i = 0; i < supervisers_.size(); ++i)
    {
        supervisers_.at(i) -> monitor(epoch);
    }
}

vf ConvolutionNeuralNetwork::getOutput() const
{
    vf output;
    for (int i = 0; i < layers_.at(layers_.size()-1)->getOutput().size(); ++i)
    {
        output.push_back(layers_.at(layers_.size()-1)->getOutput().at(i).at(0));
    }
    return output;
}

} // namespace cnn

