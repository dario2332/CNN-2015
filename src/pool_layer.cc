#include "pool_layer.h"

#include <cassert>
#include <iostream>

namespace cnn {

PoolLayer::PoolLayer(int frame_size, int num_feature_maps, int prev_map_size) 
    : Layer(prev_map_size, prev_map_size/frame_size, num_feature_maps, num_feature_maps),
      frame_size_(frame_size)
{
    assert(prev_map_size % frame_size == 0);
}

vvf& MaxPoolLayer::forwardPass(const vvf &input)
{
    //reset prev_error_
    for (int i = 0; i < num_feature_maps_; ++i)
    {
        std::fill(prev_error_.at(i).begin(), prev_error_.at(i).end(), 0);
    }

    assert(input.at(0).size() == prev_map_size_*prev_map_size_);
    assert(input.size() == num_feature_maps_);
    
    this->input_ = &input;
    
    for (int feature_map = 0; feature_map < num_feature_maps_; ++feature_map)
    {
        for (int out_row = 0; out_row < map_size_; out_row++)
        {
            for (int out_col = 0; out_col < map_size_; out_col++)
            {
                output_.at(feature_map).at(out_row*map_size_+out_col) = max(feature_map, out_row*frame_size_, out_col*frame_size_);
            }
        }
    }
    return output_;
}


float MaxPoolLayer::max(int feature_map, int in_row, int in_col)
{
    float max = input_->at(feature_map).at(in_row*prev_map_size_+in_col);
    int row = in_row, col = in_col;

    for (int i = in_row; i < in_row+frame_size_; ++i)
    {
        for (int j = in_col; j < in_col+frame_size_; ++j)
        {
            if (input_->at(feature_map).at(i*prev_map_size_ + j) > max)
            {
                max = input_->at(feature_map).at(i*prev_map_size_ + j);
                row = i; 
                col = j;
            }
        }
    }
    prev_error_.at(feature_map).at(row*prev_map_size_+col) = 1;
    return max;
}

vvf& MaxPoolLayer::backPropagate(const vvf &error)
{
    assert(error.size() == num_feature_maps_);
    assert(error.at(0).size() == map_size_*map_size_);

    for (int feature_map = 0; feature_map < num_feature_maps_; ++feature_map)
    {
        for (int in_row = 0; in_row < prev_map_size_; in_row++)
        {
            for (int in_col = 0; in_col < prev_map_size_; in_col++)
            {
                float frame_error = error.at(feature_map).at(in_row/frame_size_ * map_size_ + in_col/frame_size_);
                prev_error_.at(feature_map).at(in_row*prev_map_size_+in_col) *= frame_error;
            }
        }
    }
    return prev_error_;
}

} // namespace cnn

