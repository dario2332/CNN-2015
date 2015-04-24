#ifndef LAYER
#define LAYER
#include <vector>

typedef std::vector<float> vd;
typedef std::vector<vd> vvd;
typedef std::vector<vvd> vvvd;

class Layer
{
public:
    virtual vvd& forwardPass(const vvd &input) = 0;
    virtual vvd& backPropagate(const vvd &input) = 0;
    virtual vvd& getOutput() = 0;
    virtual vvd& getPrevError() = 0;
};


#endif
