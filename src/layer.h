#ifndef CNN2015_LAYER_H_
#define CNN2015_LAYER_H_

#include <vector>

typedef std::vector<float> vf;
typedef std::vector<vf> vvf;
typedef std::vector<vvf> vvvf;

namespace cnn {

class Layer
{
public:
    Layer(int prev_map_size, int map_size, int num_prev_feature_maps,  int num_feature_maps);
    virtual ~Layer() {};

    virtual vvf& forwardPass(const vvf &input) = 0;
    virtual vvf& backPropagate(const vvf &error) = 0;
    inline const vvf& getOutput() const { return output_; }
    inline const vvf& getPrevError() const { return prev_error_; }
    inline int getMapSize() const { return map_size_; }

protected:
    const int map_size_, prev_map_size_;
    // number of feature maps
    const int num_prev_feature_maps_, num_feature_maps_;
    const vvf *input_;
    vvf output_, prev_error_;
};

} // namespace cnn

#endif
