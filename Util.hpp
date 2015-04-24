#ifndef CNN_UTIL
#define CNN_UTIL 
#include <string>
#include <vector>
#include "layer.hpp"
#include "UtilI.hpp"
#include "ConvolutionLayer.hpp"


class ReLUInitializer : public Initializer
{
public:
    ReLUInitializer(int n) : n(n) {};
    virtual void init(vd &weights) const;
private:
    int n;
};

class TestInitializer : public Initializer
{
public:
    virtual void init(vd &weights) const;
};

class SquareCost : public CostFunction
{
public:
    SquareCost (int numOutputs) : CostFunction(numOutputs) {}
    virtual vvd& calculate(const vvd &output, const vd& expectedOutput);
};

class MnistTestInputManager : public InputManager
{
public:
    MnistTestInputManager(std::string path = "MNIST");
    virtual vvd& getTrainInput(int i) { return inputs.at(i); }
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
 *  int - learning rate
 *  float(outputFM x inputFM x kernelSize x kernelSize) - weights
******************/
class WeightRecorder : public TrainingSupervisor
{
public:
    WeightRecorder (std::vector<ConvolutionLayer*> layers, std::string datasetName) : layers(layers), datasetName(datasetName) {}
    virtual void monitor(int epoch);

private:
    std::vector<ConvolutionLayer*> layers;
    std::string datasetName;
};


#endif
