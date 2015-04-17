#ifndef CONV_LAYER
#define CONV_LAYER 
#include <vector>
#include "layer.hpp"
#include "Util.hpp"

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int mapSize, int inputFM,  int outputFM, int kernelSize, Initializer &init, float learningRate); 
    virtual vvd& forwardPass(const vvd &input);
    virtual vvd& backPropagate(const vvd &error);
    vvd& getOutput() { return output; }
    vvd& getPrevError() { return prevError; }
    vvvd& getKernel() { return kernelW; }
    void printKernel();

private:
    int mapSize, inputMapSize;
    // kernelSize x kernelSize
    int kernelSize;
    // number of feature maps
    int inputFM, outputFM;
    //kernelw[outputFM][inputFM][i*kernelSize + j]
    vvvd kernelW;
    vvd output, prevError;
    vd bias;
    const vvd *input;
    float learningRate;

    void update(const vvd &error);
    double convolve(int w, int h, const vvd &input, int outputFM);
};

class FullyConnectedLayer : public ConvolutionLayer
{
public:
    FullyConnectedLayer (int inputFM,  int outputFM, Initializer &init, float learningRate) :
                        ConvolutionLayer(1, inputFM, outputFM, 1, init, learningRate) {}
};

#endif
