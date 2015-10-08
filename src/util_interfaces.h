#ifndef CNN2015_UTILI_H_
#define CNN2015_UTILI_H_ 

#include <string>

#include "layer.h"

namespace cnn {
    
class Initializer
{
public:
    virtual void init(int n_in, int n_out, vf &weights) const = 0;
};

class CostFunction
{
public:
    explicit CostFunction(int num_outputs) : num_outputs_(num_outputs), error_(0), prev_error_(vvf(num_outputs, vf(1))) {}
    virtual ~CostFunction () {};

    virtual vvf& calculate(const vvf &output, const vf& expected_output) = 0;
    inline const vvf& getPrevError() const { return prev_error_; }
    inline float getError() const { return error_; }

protected:
    vvf prev_error_;
    float error_;
    int num_outputs_;
};

class InputManager
{
public:
    explicit InputManager (int n) : num_inputs_(n) {}
    virtual ~InputManager () {};

    virtual const vvf& getInput(int i) const = 0;
    virtual const vf& getExpectedOutput(int i) const = 0;
    inline int getInputNum() const { return num_inputs_; }
    virtual void reset() = 0;
   
protected:
    int num_inputs_;
};

class TrainingSupervisor
{
public:
    explicit TrainingSupervisor(std::string path) : path_(path) {}
    virtual ~TrainingSupervisor () {};
    
    virtual void monitor(int epoch) = 0;
protected:
    std::string path_;
};

} // namespace cnn

#endif

