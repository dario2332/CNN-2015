#ifndef CNN
#define CNN
#include "Util.hpp"
#include "layer.hpp"

class ConvolutionNeuralNetwork
{
public:
    ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, const CostFunction &costFunction, const InputManager &inputManager);
    void forwardPass();
    void backPropagate();
    void train();
    
private:
    std::vector<Layer*> layers;
    CostFunction &costFunction;
    InputManager &inputManager;
};


#endif
