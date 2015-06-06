#ifndef POOL_LAYER
#define POOL_LAYER value
#include "layer.hpp"


class PoolLayer : public Layer
{
public:
    PoolLayer (int frameSize, int numFM, int prevMapSize); 

protected:
    const int frameSize;
};

class MaxPoolLayer : public PoolLayer
{
public:
    MaxPoolLayer(int frameSize, int numFM, int inMapSize) : PoolLayer(frameSize, numFM, inMapSize) {}
    virtual vvf& forwardPass(const vvf &input);
    virtual vvf& backPropagate(const vvf &error);

private:
    float max(int fm, int iRow, int iCol);

};

#endif

