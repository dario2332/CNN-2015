#ifndef CNN2015_ACTIVATION_LAYER_H_
#define CNN2015_ACTIVATION_LAYER_H_

#include "layer.h"

namespace cnn {

class ActivationLayer : public Layer
{
public:
    explicit ActivationLayer(int num_feature_maps, int map_size = 1);

    vvf& forwardPass(const vvf &input) override;
    vvf& backPropagate(const vvf &error) override;

protected:
    virtual float activationFunction(float x) = 0;
    virtual float activationFunctionDerivative(float x) = 0;
};

class TanhLayer : public ActivationLayer 
{
public:
    explicit TanhLayer(int num_feature_maps, int map_size = 1) : ActivationLayer(num_feature_maps, map_size) {}

protected:
    virtual float activationFunction(float x) override;
    virtual float activationFunctionDerivative(float x) override;
};

} // namespace cnn
#endif
