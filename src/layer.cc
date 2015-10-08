#include "layer.h"

namespace cnn {

Layer::Layer(int prev_map_size, int map_size, int num_prev_feature_maps,  int num_feature_maps) 
    : prev_map_size_(prev_map_size), map_size_(map_size), num_prev_feature_maps_(num_prev_feature_maps),
      num_feature_maps_(num_feature_maps), output_(vvf(num_feature_maps, vf(map_size*map_size))), 
      prev_error_(vvf(num_prev_feature_maps, vf(prev_map_size*prev_map_size, 0))) {}

} // namespace cnn

