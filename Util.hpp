#ifndef CNN_UTIL
#define CNN_UTIL 
#include <string>
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

class MnistSmallInputManager : public InputManager
{
public:
    MnistSmallInputManager(std::string path = "MNIST");
    virtual vvd& getInput(int i) { return inputs.at(i); }
    virtual vd& getExpectedOutput(int i) { return expectedOutputs.at(i); }
    virtual void reset() {}

private:
    vvvd inputs;
    vvd expectedOutputs;
};

/*****************
 * Record all weights from convolution layers in files.
 *  int - number of output feature maps (1. dimension of kernel)
 *  int - number of input feature maps (2. dimension of kernel)
 *  int - kernel size
 *  int - output map size
 *  float(outputFM x inputFM x kernelSize x kernelSize) - weights
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

#endif
