#ifndef UTIL_I
#define UTIL_I 
#include "layer.hpp"

class Initializer
{
public:
    virtual void init(vd &weights) const = 0;
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
    InputManager (int train, int validate, int test) : train(train), test(test), validate(validate) {}
    virtual vvd& getTrainInput(int i) = 0;
    virtual vd& getExpectedOutput(int i) = 0;
    int getTrainNum() { return train; }
    int getTestNum() { return test; }
    int getValNum() { return validate; }
    virtual void reset() = 0;
    //virtual void preProcess() = 0;
   
private:
    // number of images in each set
    int train, test, validate;
};

class TrainingSupervisor
{
public:
    virtual void monitor(int epoch) = 0;
};


#endif
