#ifndef CNN2015_UTIL_H_
#define CNN2015_UTIL_H_ 

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "convolution_neural_network.h"
#include "convolution_layer.h"
#include "layer.h"
#include "util_interfaces.h"

class ConvolutionNeuralNetwork;

namespace cnn {

class ReLUInitializer : public Initializer
{
public:
    explicit ReLUInitializer(int n) : n_(n) {};

    void init(int n_in, int n_out, vf &weights) const override;
private:
    int n_;
};

class TestInitializer : public Initializer
{
public:
    void init(int n_in, int n_out, vf &weights) const override;
};

class TanhInitializer : public Initializer
{
public:
    void init(int n_in, int n_out, vf &weights) const override;
};

class SquareCost : public CostFunction
{
public:
    explicit SquareCost (int num_outputs) : CostFunction(num_outputs) {}

    vvf& calculate(const vvf &output, const vf& expected_output) override;
};

class MnistInputManager : public InputManager
{
public:
    explicit MnistInputManager (int num, std::string path = "MNIST");    

    inline const vvf& getInput(int i) const override { return inputs_.at(indexes_.at(i)); }
    inline const vf& getExpectedOutput(int i) const override { return expected_outputs_.at(indexes_.at(i)); }
    void reset() override { std::random_shuffle( indexes_.begin(), indexes_.end()); }

protected:
    void preprocess();
    void readData(std::ifstream &in_images, std::ifstream &in_labels);  

    vvvf inputs_;
    vvf expected_outputs_;
    std::vector<int> indexes_;
};

class MnistSmallInputManager : public MnistInputManager
{
public:
    explicit MnistSmallInputManager(std::string path = "MNIST");
};

class MnistTrainInputManager : public MnistInputManager
{
public:
    explicit MnistTrainInputManager(std::string path = "MNIST");
};

class MnistValidateInputManager : public MnistInputManager
{
public:
    explicit MnistValidateInputManager(std::string path = "MNIST");
};

class MnistTestInputManager : public MnistInputManager
{
public:
    explicit MnistTestInputManager(std::string path = "MNIST");
};

/*****************
 * Record all weights from convolution layers in files.
 *  int - number of output feature maps (1. dimension of kernel)
 *  int - number of input feature maps (2. dimension of kernel)
 *  int - kernel size
 *  int - output map size
 *  float(outputFM x inputFM x kernelSize x kernelSize) - weights
 *  float(outputFM) - bias
******************/
class WeightRecorder : public TrainingSupervisor
{
public:
    WeightRecorder (const std::vector<ConvolutionLayer*> layers, std::string path) :TrainingSupervisor(path), layers_(layers) {}

    void monitor(int epoch) override;

private:
    std::vector<ConvolutionLayer*> layers_;
};

class Validator : public TrainingSupervisor
{
public:
    Validator (ConvolutionNeuralNetwork &cnn, InputManager& input_manager, std::string path, int num_classes);

    void monitor(int epoch) override;

private:
    InputManager &input_manager_;
    ConvolutionNeuralNetwork &cnn_;
    vvf possible_outputs_;
};

class ActivationVariance : public TrainingSupervisor
{
public:
    ActivationVariance (std::vector<Layer*> layers, std::string path) : TrainingSupervisor(path), layers_(layers) {}

    void monitor(int epoch) override;

private:
    std::vector<Layer*> layers_;
};

class GradientVariance : public TrainingSupervisor
{
public:
    GradientVariance (std::vector<Layer*> layers, std::string path) : TrainingSupervisor(path), layers_(layers) {}

    void monitor(int epoch) override;

private:
    std::vector<Layer*> layers_;
};

} // namespace cnn

#endif

