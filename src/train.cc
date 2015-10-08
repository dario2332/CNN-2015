#include <iostream>
#include <cassert>

#include "activation_layer.h"
#include "convolution_neural_network.h"
#include "convolution_layer.h"
#include "layer.h"
#include "pool_layer.h"
#include "util.h"
#include "util_interfaces.h"

void trainMnist(int epoch)
{
    std::vector<cnn::Layer*> layers;
    std::vector<cnn::ConvolutionLayer*> conv_layers;
    cnn::MnistTrainInputManager minst_input;
    cnn::MnistValidateInputManager validate_mnist;
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

    cnn::ConvolutionNeuralNetwork cnn(layers, sc, minst_input);

    conv_layers.push_back(&conv1);
    conv_layers.push_back(&conv2);
    conv_layers.push_back(&conv3);
    conv_layers.push_back(&full1);
    conv_layers.push_back(&full2);

    cnn::WeightRecorder wr(conv_layers, "MNIST");
    /********************************
     * Supervisors for training
     *********************************
    cnn::Validator train_validator(cnn, minst_input, "MNIST/train", 10);
    cnn::Validator validate_validator(cnn, validate_mnist, "MNIST/validate", 10);
    cnn::ActivationVariance avar(layers, "MNIST/");
    cnn::GradientVariance gvar(layers, "MNIST/");

    cnn.registerSupervisor(&avar);
    cnn.registerSupervisor(&gvar);
    cnn.registerSupervisor(&train_validator);
    cnn.registerSupervisor(&validate_validator);
    cnn.registerSupervisor(&wr);
    ********************************************/
    
    //train for 10 epochs
    cnn.train(epoch);
    //for recording weights
    wr.monitor(0);
}

int main(int argc, char *argv[])
{
    trainMnist(std::atoi(argv[1]));
    return 0;
}
