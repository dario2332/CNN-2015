#ifndef POOL_LAYER
#define POOL_LAYER value
#include "layer.hpp"


class PoolLayer : public Layer
{
public:
    PoolLayer (int kernelSize, int numFM, int inputMapSize); 
    virtual vvd& forwardPass(const vvd &input) = 0;
    virtual vvd& backPropagate(const vvd &error) = 0;
    vvd& getPrevError() { return prevError; }
    vvd& getOutput() { return output; }
protected:
    int kernelSize;
    const vvd *input;
    const int numFM, inMapSize, outMapSize;
    vvd output, prevError;


};

class MaxPoolLayer : public PoolLayer
{
public:
    MaxPoolLayer(int kernelSize, int numFM, int inMapSize) : PoolLayer(kernelSize, numFM, inMapSize) {}
    virtual vvd& forwardPass(const vvd &input);
    virtual vvd& backPropagate(const vvd &error);

private:
    float max(int fm, int iRow, int iCol);

};

#endif

