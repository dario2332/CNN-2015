#ifndef UTIL_I
#define UTIL_I 
#include "layer.hpp"
#include <string>

class Initializer
{
public:
    virtual void init(vd &weights, int n_in, int n_out) const = 0;
};

class CostFunction
{
public:
    CostFunction(int numOutputs) : numOutputs(numOutputs), error(0), prevError(vvd(numOutputs, vd(1))) {}
    virtual vvd& calculate(const vvd &output, const vd& expectedOutput) = 0;
    vvd& getPrevError() { return prevError; }
    float getError() { return error; }

protected:
    vvd prevError;
    float error;
    int numOutputs;
};

class InputManager
{
public:
    InputManager (int n) : numOfInputs(n) {}
    virtual vvd& getInput(int i) = 0;
    virtual vd& getExpectedOutput(int i) = 0;
    int getInputNum() { return numOfInputs; }
    virtual void reset() = 0;
    //virtual void preProcess() = 0;
   
protected:
    // number of images in each set
    int numOfInputs;
};

class TrainingSupervisor
{
public:
    TrainingSupervisor(std::string path) : path(path) {}
    virtual void monitor(int epoch = 0) = 0;
protected:
    std::string path;
};


#endif
