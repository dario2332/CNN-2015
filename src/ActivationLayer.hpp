#ifndef ACTIVATION_LAYER
#define ACTIVATION_LAYER
#include "layer.hpp"

class ActivationLayer : public Layer
{
public:
    ActivationLayer(int numFM, int mapSize = 1);
    virtual vvf& forwardPass(const vvf &input);
    virtual vvf& backPropagate(const vvf &error);

protected:
    virtual float activationFunction(float x) = 0;
    virtual float activationFunctionDerivative(float x) = 0;
};

class TanhLayer : public ActivationLayer 
{
public:
    TanhLayer(int numFM, int mapSize = 1) : ActivationLayer(numFM, mapSize) {}

protected:
    virtual float activationFunction(float x);
    virtual float activationFunctionDerivative(float x);
};

#endif
