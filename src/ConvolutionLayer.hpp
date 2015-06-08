#ifndef CONV_LAYER
#define CONV_LAYER 
#include <vector>
#include "layer.hpp"
#include "UtilI.hpp"

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int mapSize, int prevFM,  int numFM, int kernelSize, Initializer &init, float learningRate); 
    virtual vvf& forwardPass(const vvf &input);
    virtual vvf& backPropagate(const vvf &error);
    vvvf& getKernel() { return kernelW; }
    vf& getBias() { return bias; }
    float getLearningRate() { return learningRate; }
    void printKernel();
    void writeKernel(std::string path);
    void loadWeights(std::string file);

private:
    const int kernelSize;
    //kernelw[numFM][prevFM][i*kernelSize + j]
    vvvf kernelW;
    vf bias;
    float learningRate;

    void update(const vvf &error);
    double convolve(int w, int h, const vvf &input, int numFM);
};

class FullyConnectedLayer : public ConvolutionLayer
{
public:
    FullyConnectedLayer (int prevFM,  int numFM, Initializer &init, float learningRate) :
                        ConvolutionLayer(1, prevFM, numFM, 1, init, learningRate) {}
};

#endif
