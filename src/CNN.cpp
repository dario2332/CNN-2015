#include "CNN.hpp"


ConvolutionNeuralNetwork::ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, 
                                                    CostFunction &costFunction, 
                                                    InputManager &inputManager) :
                                                    layers(layers),
                                                    costFunction(costFunction),
                                                    inputManager(inputManager) {}

void ConvolutionNeuralNetwork::feedForward(vvf &input)
{
    vvf *currentInput = &input;
    for (int layer = 0; layer < layers.size(); ++layer)
    {
        layers.at(layer)->forwardPass(*currentInput);
        currentInput = &(layers.at(layer)->getOutput());
    }
}

void ConvolutionNeuralNetwork::backPropagate(vvf &error)
{
    vvf *currentError = &error;
    for (int layer = layers.size() - 1; layer >= 0; --layer)
    {
        layers.at(layer)->backPropagate(*currentError);
        currentError = &layers.at(layer)->getPrevError();
    }
}

void ConvolutionNeuralNetwork::train(int numEpochs)
{
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        for (int i = 0, n = inputManager.getInputNum(); i < n; ++i)
        {
            feedForward(inputManager.getInput(i));
            backPropagate(costFunction.calculate(layers.at(layers.size()-1)->getOutput(), inputManager.getExpectedOutput(i)));
        }
        notifySupervisors(epoch);
        inputManager.reset();
    }
}

float ConvolutionNeuralNetwork::getCost(vf &expectedOutput)
{
    costFunction.calculate(layers.at(layers.size()-1)->getOutput(), expectedOutput);
    return costFunction.getError();
}

void ConvolutionNeuralNetwork::notifySupervisors(int epoch)
{
    for (int i = 0; i < supervisers.size(); ++i)
    {
        supervisers.at(i) -> monitor(epoch);
    }
}

