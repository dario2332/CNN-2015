#ifndef UTIL_I
#define UTIL_I 
#include "layer.hpp"
#include <string>

class Initializer
{
public:
    virtual void init(vf &weights, int n_in, int n_out) const = 0;
};

class CostFunction
{
public:
    CostFunction(int numOutputs) : numOutputs(numOutputs), error(0), prevError(vvf(numOutputs, vf(1))) {}
    virtual vvf& calculate(const vvf &output, const vf& expectedOutput) = 0;
    vvf& getPrevError() { return prevError; }
    float getError() { return error; }

protected:
    vvf prevError;
    float error;
    int numOutputs;
};

class InputManager
{
public:
    InputManager (int n) : numOfInputs(n) {}
    virtual vvf& getInput(int i) = 0;
    virtual vf& getExpectedOutput(int i) = 0;
    int getInputNum() { return numOfInputs; }
    virtual void reset() = 0;
   
protected:
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
