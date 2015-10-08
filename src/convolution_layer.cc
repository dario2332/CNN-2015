#include "convolution_layer.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <cblas.h>
#include <opencv2/opencv.hpp>

namespace cnn {

ConvolutionLayer::ConvolutionLayer(int map_size, int num_prev_feature_maps, int num_feature_maps, 
                                   int kernel_size, const Initializer &init, float learning_rate) 
    : Layer(map_size + kernel_size - 1, map_size, num_prev_feature_maps, num_feature_maps),  
      kernel_size_(kernel_size), learning_rate_(learning_rate),
      bias_(vf(num_feature_maps, 0)), 
      kernel_(vvvf(num_feature_maps, vvf(num_prev_feature_maps, vf(kernel_size*kernel_size))))
{
    for (int out_feature_map = 0; out_feature_map < num_feature_maps; ++out_feature_map)
    {
        for (int in_feature_map = 0; in_feature_map < num_prev_feature_maps; ++in_feature_map)
        {
            int n_in = num_prev_feature_maps*num_prev_feature_maps*num_prev_feature_maps;
            int n_out = num_feature_maps * map_size * map_size;
            init.init(n_in, n_out, kernel_.at(out_feature_map).at(in_feature_map));
        }
    }
}    

vvf& ConvolutionLayer::forwardPass(const vvf &input) 
{
    assert(input.size() == num_prev_feature_maps_);
    assert(input.at(0).size() == prev_map_size_ * prev_map_size_);

    this->input_ = &input;

    for (int feature_map = 0, o = output_.size(); feature_map < o; ++feature_map)
    {
        for (int row = 0; row < map_size_; ++row)
        {
            for (int col = 0; col < map_size_; ++col)
            {
                output_.at(feature_map).at(row*map_size_ + col) = convolve(row, col, input, feature_map);
                output_.at(feature_map).at(row*map_size_ + col) += bias_.at(feature_map);
            }
        }
    }
    
    //reset prev_error_ to 0
    for (int i = 0; i < num_prev_feature_maps_; ++i)
    {
        std::fill(prev_error_.at(i).begin(), prev_error_.at(i).end(), 0);
    }
    return output_;
}

double ConvolutionLayer::convolve(int row, int col, const vvf &input, int out_feature_map)
{
    int input_size = sqrt(input.at(0).size()); 
    assert(input_size == map_size_ + kernel_size_ - 1);

    double result = 0;
    for (int i = 0; i < num_prev_feature_maps_; ++i)
    {
        for (int j = row; j < row+kernel_size_; ++j)
        {
            const float *input_area = &input.at(i).at(j*input_size + col);
            const float *kernel_area = &kernel_.at(out_feature_map).at(i).at((j-row)*kernel_size_);
            // dot product
            result += cblas_dsdot(kernel_size_, input_area, 1, kernel_area, 1);
        }
    }
    return result;
}

vvf& ConvolutionLayer::backPropagate(const vvf &error) 
{
    assert(error.size() == num_feature_maps_);
    assert(error.at(0).size() == map_size_ * map_size_);

    for (int out_feature_map = 0; out_feature_map < num_feature_maps_; ++out_feature_map)
    {
        for (int out_row = 0; out_row < map_size_; ++out_row)
        {
            for (int in_feature_map = 0; in_feature_map < num_prev_feature_maps_; ++in_feature_map)
            {
                for (int in_row = out_row; in_row < out_row + kernel_size_; ++in_row)
                {
                    for (int out_col = 0; out_col < map_size_; ++out_col)
                    {
                        float err = error.at(out_feature_map).at(out_row*map_size_ + out_col);
                        float *kernel_area = &kernel_.at(out_feature_map).at(in_feature_map).at((in_row-out_row)*kernel_size_);
                        float *prev_error_area = &prev_error_.at(in_feature_map).at(in_row*prev_map_size_ + out_col);
                        // prev_error_area = err*kernel_area + prev_error_area
                        cblas_saxpy(kernel_size_, err, kernel_area, 1, prev_error_area, 1);
                    }
                }
            }
        }
    }
    update(error);
    return prev_error_;
}

void ConvolutionLayer::update(const vvf &error)
{
    for (int out_feature_map = 0; out_feature_map < num_feature_maps_; ++out_feature_map)
    {
        for (int in_feature_map = 0; in_feature_map < num_prev_feature_maps_; ++in_feature_map)
        {
            for (int kernel_row = 0; kernel_row < kernel_size_; ++kernel_row)
            {
                for (int kernel_col = 0; kernel_col < kernel_size_; ++kernel_col)
                {
                    for (int in_row = kernel_row; in_row < kernel_row+map_size_; ++in_row)
                    {
                        const float *input_area = &(input_->at(in_feature_map).at(in_row*prev_map_size_+kernel_col));
                        const float *error_area = &error.at(out_feature_map).at((in_row-kernel_row)*map_size_);

                        float update = learning_rate_ * cblas_sdot(map_size_, input_area, 1, error_area, 1);
                        kernel_.at(out_feature_map).at(in_feature_map).at(kernel_row*kernel_size_ + kernel_col) -= update;
                    }
                }
            }
        }
        bias_.at(out_feature_map) -= learning_rate_ * std::accumulate(error.at(out_feature_map).begin(), 
                                                                      error.at(out_feature_map).end(), 0);
    }
}

void ConvolutionLayer::printKernel() const
{
    for (int out_feature_map = 0; out_feature_map < num_feature_maps_; ++out_feature_map)
    {
        for (int in_feature_map = 0; in_feature_map < num_prev_feature_maps_; ++in_feature_map)
        {
            for (int kernel_row = 0; kernel_row < kernel_size_; ++kernel_row)
            {
                for (int kernel_col = 0; kernel_col < kernel_size_; ++kernel_col)
                {
                    std::cout << kernel_.at(out_feature_map).at(in_feature_map).at(kernel_row*kernel_size_+kernel_col) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void ConvolutionLayer::loadWeights(std::string file)
{
    std::ifstream in(file, std::fstream::binary);
    int out_feature_map, in_feature_map, kernel_size;
    int out_map_size;
    
    in.read(reinterpret_cast<char*>(&out_feature_map), sizeof(out_feature_map));
    in.read(reinterpret_cast<char*>(&in_feature_map), sizeof(in_feature_map));
    in.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
    in.read(reinterpret_cast<char*>(&out_map_size), sizeof(out_map_size));
    
    assert(out_feature_map == num_feature_maps_);
    assert(in_feature_map == num_prev_feature_maps_);
    assert(kernel_size == this->kernel_size_);
    assert(out_map_size == map_size_);

    for (out_feature_map = 0; out_feature_map < num_feature_maps_; ++out_feature_map)
    {
        for (in_feature_map = 0; in_feature_map < num_prev_feature_maps_; ++in_feature_map)
        {
            for (int k = 0; k < kernel_.at(0).at(0).size(); ++k)
            {
                in.read(reinterpret_cast<char*>(&kernel_.at(out_feature_map).at(in_feature_map).at(k)), sizeof(float));
            }
        }
    }
    for (int i = 0; i < bias_.size(); ++i)
    {
        in.read(reinterpret_cast<char*>(&bias_.at(i)), sizeof(float));
    }
    in.close();
}

void ConvolutionLayer::writeKernel(std::string path) const
{
    cv::Mat image(kernel_size_, kernel_size_, CV_8UC1);
    for (int out_feature_map = 0; out_feature_map < num_feature_maps_; ++out_feature_map)
    {
        for (int in_feature_map = 0; in_feature_map < num_prev_feature_maps_; ++in_feature_map)
        {
            for (int k = 0; k < kernel_size_ * kernel_size_; ++k)
            {
                image.at<unsigned char>(k/kernel_size_, k%kernel_size_) = kernel_.at(out_feature_map).at(in_feature_map).at(k);
            }
            cv::imwrite(path + "Kernel" + std::to_string(out_feature_map) + "_" + std::to_string(in_feature_map) + ".jpg", image);
        }
    }
}

} // namespace cnn

