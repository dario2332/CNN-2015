#ifndef CNN2015_CONVOLUTION_NEURAL_NETWORK_H_
#define CNN2015_CONVOLUTION_NEURAL_NETWORK_H_

#include "layer.h"
#include "util_interfaces.h"

namespace cnn {

class ConvolutionNeuralNetwork
{
public:
    ConvolutionNeuralNetwork (const std::vector<Layer*> &layers, CostFunction &cost_function, InputManager &input_manager);

    void feedForward(const vvf &input);
    void backPropagate(const vvf &error);
    void train(int num_epochs);
    inline void registerSupervisor(TrainingSupervisor *s) { supervisers_.push_back(s); }
    void notifySupervisors(int epoch) const;
    float getCost(const vf &expected_output) const;
    inline vf getOutput() const;
    inline InputManager& getInputManager() const { return input_manager_; }
    
private:
    std::vector<Layer*> layers_;
    CostFunction &cost_function_;
    InputManager &input_manager_;
    std::vector<TrainingSupervisor*> supervisers_;
};

} // namespace cnn

#endif
