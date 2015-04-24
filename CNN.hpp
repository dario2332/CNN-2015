#ifndef CNN
#define CNN
#include "Util.hpp"
#include "layer.hpp"

class ConvolutionNeuralNetwork
{
public:
    ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, CostFunction &costFunction, InputManager &inputManager);
    void forwardPass(vvd &input);
    void backPropagate(vvd &error);
    void train(int numEpochs, bool supervized = false);
    void registerSupervisor(TrainingSupervisor *s) { supervisers.push_back(s); }
    void notifySupervisors(int epoch);
    
private:
    std::vector<Layer*> layers;
    CostFunction &costFunction;
    InputManager &inputManager;
    std::vector<TrainingSupervisor*> supervisers;
};


#endif
