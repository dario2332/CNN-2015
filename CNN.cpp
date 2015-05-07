#include "CNN.hpp"


ConvolutionNeuralNetwork::ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, 
                                                    CostFunction &costFunction, 
                                                    InputManager &inputManager) :
                                                    layers(layers),
                                                    costFunction(costFunction),
                                                    inputManager(inputManager) {}

void ConvolutionNeuralNetwork::forwardPass(vvd &input)
{
    vvd *currentInput = &input;
    for (int layer = 0; layer < layers.size(); ++layer)
    {
        layers.at(layer)->forwardPass(*currentInput);
        currentInput = &(layers.at(layer)->getOutput());
    }
}

void ConvolutionNeuralNetwork::backPropagate(vvd &error)
{
    vvd *currentError = &error;
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
            forwardPass(inputManager.getInput(i));
            backPropagate(costFunction.calculate(layers.at(layers.size()-1)->getOutput(), inputManager.getExpectedOutput(i)));
        }
        notifySupervisors(epoch);
            // proci kroz validation test i ispisati tocnost
            // takoder tocnost training seta
            // updates of weights
            // gradient check jos nemam u konvolucijskim i aktivacijskim slojevima
            // varijanca aktivacija i gradijenata svakog sloja
        inputManager.reset();
    }
}

float ConvolutionNeuralNetwork::getCost(vd &expectedOutput)
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

