#include <iostream>
#include <cassert>

#include "activation_layer.h"
#include "convolution_neural_network.h"
#include "convolution_layer.h"
#include "layer.h"
#include "pool_layer.h"
#include "util.h"
#include "util_interfaces.h"

void testMnist()
{
    std::vector<cnn::Layer*> layers;
    cnn::MnistValidateInputManager validate_mnist;
    cnn::MnistTestInputManager test_mnist;
    cnn::SquareCost sc(10);
    cnn::TanhInitializer init;
    const float learning_rate = 0.001;
    
    cnn::ConvolutionLayer conv1(28, 1, 6, 5, init, learning_rate);
    cnn::TanhLayer tanh1(6, 28);
    cnn::MaxPoolLayer pool1(2, 6, 28);
    cnn::ConvolutionLayer conv2(10, 6, 16, 5, init, learning_rate);
    cnn::TanhLayer tanh2(16, 10);
    cnn::MaxPoolLayer pool2(2, 16, 10);
    cnn::ConvolutionLayer conv3(1, 16, 100, 5, init, learning_rate);
    cnn::TanhLayer tanh3(100, 1);
    cnn::FullyConnectedLayer full1(100, 80, init, learning_rate);
    cnn::TanhLayer tanh_full1(80);
    cnn::FullyConnectedLayer full2(80, 10, init, learning_rate);
    
    layers.push_back(&conv1);
    layers.push_back(&tanh1);
    layers.push_back(&pool1);
    layers.push_back(&conv2);
    layers.push_back(&tanh2);
    layers.push_back(&pool2);
    layers.push_back(&conv3);
    layers.push_back(&tanh3);
    layers.push_back(&full1);
    layers.push_back(&tanh_full1);
    layers.push_back(&full2);

    cnn::ConvolutionNeuralNetwork cnn(layers, sc, test_mnist);
    cnn::Validator test(cnn, test_mnist, "MNIST/test", 10);

    conv1.loadWeights("MNIST/Weights_E0_CL1");
    conv2.loadWeights("MNIST/Weights_E0_CL2");
    conv3.loadWeights("MNIST/Weights_E0_CL3");
    full1.loadWeights("MNIST/Weights_E0_CL4");
    full2.loadWeights("MNIST/Weights_E0_CL5");

    test.monitor(0);
}

int main(int argc, char *argv[])
{
    testMnist();
    return 0;
}
