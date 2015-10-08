#ifndef CNN2015_CONVOLUTION_LAYER_H_
#define CNN2015_CONVOLUTION_LAYER_H_

#include <string>

#include "layer.h"
#include "util_interfaces.h"

namespace cnn {

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int map_size, int num_prev_feature_maps,  int num_feature_maps, 
                     int kernel_size, const Initializer &init, float learning_rate); 

    vvf& forwardPass(const vvf &input) override;
    vvf& backPropagate(const vvf &error) override;
    inline const vvvf& getKernel() const { return kernel_; }
    inline const vf& getBias() const { return bias_; }
    inline float getLearningRate() const { return learning_rate_; }
    void printKernel() const;
    void writeKernel(std::string path) const;
    void loadWeights(std::string file);

private:
    void update(const vvf &error);
    double convolve(int w, int h, const vvf &input, int feature_map);

    const int kernel_size_;
    //kernelw[num_feature_maps_][num_prev_feature_maps_][i*kernel_size_ + j]
    vvvf kernel_;
    vf bias_;
    float learning_rate_;
};

class FullyConnectedLayer : public ConvolutionLayer
{
public:
    FullyConnectedLayer (int num_prev_feature_maps,  int num_feature_maps, const Initializer &init, float learning_rate)
        : ConvolutionLayer(1, num_prev_feature_maps, num_feature_maps, 1, init, learning_rate) {}
};

} // namespace cnn

#endif
