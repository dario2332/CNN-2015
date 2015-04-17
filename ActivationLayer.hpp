#ifndef ACTIVATION_LAYER
#define ACTIVATION_LAYER
#include "layer.hpp"

class ActivationLayer : public Layer
{
public:
    ActivationLayer(int numFM, int mapSize = 1);
    virtual vvd& forwardPass(const vvd &input) = 0;
    virtual vvd& backPropagate(const vvd &error) = 0;
    vvd& getOutput() { return output; }
    vvd& getPrevError() { return prevError; }
    
protected:
    int numFM, mapSize;
    vvd prevError, output;
    const vvd *input;
};

class SigmoidLayer : public ActivationLayer 
{
public:
    SigmoidLayer(int numFM, int mapSize = 1) : ActivationLayer(numFM, mapSize) {}
    virtual vvd& forwardPass(const vvd &input);
    virtual vvd& backPropagate(const vvd &error);
private:
    static float sigmoid(float x);
    static float sigmoidDerivative(float x);
};

#endif
