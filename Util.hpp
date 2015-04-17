#ifndef CNN_UTIL
#define CNN_UTIL 
#include <string>
#include <vector>
#include "layer.hpp"


class Initializer
{
public:
    virtual void init(vd &weights) const = 0;
};

class ReLUInitializer : public Initializer
{
public:
    ReLUInitializer(int n) : n(n) {};
    virtual void init(vd &weights) const;
private:
    int n;
};

class TestInitializer : public Initializer
{
public:
    virtual void init(vd &weights) const;
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

class SquareCost : public CostFunction
{
public:
    SquareCost (int numOutputs) : CostFunction(numOutputs) {}
    virtual vvd& calculate(const vvd &output, const vd& expectedOutput);
};

class InputManager
{
public:
    InputManager (std::string dataset);
    virtual vvd getNextInput() = 0;
    virtual void preProcess() = 0;
};

#endif
