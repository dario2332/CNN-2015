#ifndef CNN_UTIL
#define CNN_UTIL 
#include <string>
#include <algorithm>
#include <vector>
#include "layer.hpp"
#include "UtilI.hpp"
#include "ConvolutionLayer.hpp"
#include "CNN.hpp"

class ConvolutionNeuralNetwork;

class ReLUInitializer : public Initializer
{
public:
    ReLUInitializer(int n) : n(n) {};
    virtual void init(vd &weights, int n_in, int n_out) const;
private:
    int n;
};

class TestInitializer : public Initializer
{
public:
    virtual void init(vd &weights, int n_in, int n_out) const;
};

class SigmoidInitializer : public Initializer
{
public:
    virtual void init(vd &weights, int n_in, int n_out) const;
};

class SquareCost : public CostFunction
{
public:
    SquareCost (int numOutputs) : CostFunction(numOutputs) {}
    virtual vvd& calculate(const vvd &output, const vd& expectedOutput);
};

class MnistInputManager : public InputManager
{
public:
    MnistInputManager (int num, std::string path = "MNIST");    
    virtual vvd& getInput(int i) { return inputs.at(indexes.at(i)); }
    virtual vd& getExpectedOutput(int i) { return expectedOutputs.at(indexes.at(i)); }
    virtual void reset() { std::random_shuffle( indexes.begin(), indexes.end()); }

protected:
    vvvd inputs;
    vvd expectedOutputs;
    std::vector<int> indexes;
    virtual void preprocess();
};

class MnistSmallInputManager : public MnistInputManager
{
public:
    MnistSmallInputManager(std::string path = "MNIST");

};

class MnistTrainInputManager : public MnistInputManager
{
public:
    MnistTrainInputManager(std::string path = "MNIST");

};

class MnistValidateInputManager : public MnistInputManager
{
public:
    MnistValidateInputManager(std::string path = "MNIST");

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
    WeightRecorder (std::vector<ConvolutionLayer*> layers, std::string path) :TrainingSupervisor(path), layers(layers) {}
    virtual void monitor(int epoch);

private:
    std::vector<ConvolutionLayer*> layers;
};

class Validator : public TrainingSupervisor
{
public:
    Validator (ConvolutionNeuralNetwork &cnn, InputManager& im, std::string path, int numClasses);
    virtual void monitor(int epoch);

private:
    InputManager &im;
    ConvolutionNeuralNetwork &cnn;
    vvd possibleOutputs;

};

class ActivationVariance : public TrainingSupervisor
{
public:
    ActivationVariance (std::vector<Layer*> layers, std::string path) : TrainingSupervisor(path), layers(layers) {}
    virtual void monitor(int epoch);

private:
    std::vector<Layer*> layers;
};

class GradientVariance : public TrainingSupervisor
{
public:
    GradientVariance (std::vector<Layer*> layers, std::string path) : TrainingSupervisor(path), layers(layers) {}
    virtual void monitor(int epoch);

private:
    std::vector<Layer*> layers;
};

#endif
