#ifndef CNN
#define CNN
#include "Util.hpp"
#include "layer.hpp"

class ConvolutionNeuralNetwork
{
public:
    ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, CostFunction &costFunction, InputManager &inputManager);
    void feedForward(vvf &input);
    void backPropagate(vvf &error);
    void train(int numEpochs);
    void registerSupervisor(TrainingSupervisor *s) { supervisers.push_back(s); }
    void notifySupervisors(int epoch);
    float getCost(vf &expectedOutput);
    InputManager& getInputManager() { return inputManager; }
    
private:
    std::vector<Layer*> layers;
    CostFunction &costFunction;
    InputManager &inputManager;
    std::vector<TrainingSupervisor*> supervisers;
};


#endif
