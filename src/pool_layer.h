#ifndef CNN2015_POOL_LAYER_H_
#define CNN2015_POOL_LAYER_H_

#include "layer.h"

namespace cnn {

class PoolLayer : public Layer
{
public:
    PoolLayer (int frame_size, int num_feature_maps, int prev_map_size); 

protected:
    const int frame_size_;
};

class MaxPoolLayer : public PoolLayer
{
public:
    MaxPoolLayer(int frame_size, int num_feature_maps, int prev_map_size) : PoolLayer(frame_size, num_feature_maps, prev_map_size) {}

    vvf& forwardPass(const vvf &input) override;
    vvf& backPropagate(const vvf &error) override;

private:
    float max(int feature_map, int in_row, int in_col);
};

} // namespace cnn

#endif

