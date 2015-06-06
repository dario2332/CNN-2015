#ifndef LAYER
#define LAYER
#include <vector>

typedef std::vector<float> vf;
typedef std::vector<vf> vvf;
typedef std::vector<vvf> vvvf;

class Layer
{
public:
    Layer(int prevMapSize, int mapSize, int prevFM,  int numFM);
    virtual vvf& forwardPass(const vvf &input) = 0;
    virtual vvf& backPropagate(const vvf &error) = 0;
    vvf& getOutput() { return output; }
    vvf& getPrevError() { return prevError; }
    int getMapSize() { return mapSize; }

protected:
    const int mapSize, prevMapSize;
    // number of feature maps
    const int prevFM, numFM;
    vvf output, prevError;
    const vvf *input;

};


#endif
