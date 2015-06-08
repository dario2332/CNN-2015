#include <iostream>
#include <unistd.h>
#include <cassert>
#include "CNN.hpp"
#include "layer.hpp"
#include "Util.hpp"
#include "UtilI.hpp"
#include "ConvolutionLayer.hpp"
#include "PoolLayer.hpp"
#include "ActivationLayer.hpp"


void trainMnist()
{
    std::vector<Layer*> layers;
    std::vector<ConvolutionLayer*> convLayers;
    MnistTrainInputManager minstInput;
    MnistValidateInputManager validateMnist;
    SquareCost sc(10);
    const float learningRate = 0.001;

    //TestInitializer init;
    TanhInitializer init;
    
    ConvolutionLayer conv1(28, 1, 6, 5, init, learningRate);
    TanhLayer tanh1(6, 28);
    MaxPoolLayer pool1(2, 6, 28);
    ConvolutionLayer conv2(10, 6, 16, 5, init, learningRate);
    TanhLayer tanh2(16, 10);
    MaxPoolLayer pool2(2, 16, 10);
    ConvolutionLayer conv3(1, 16, 100, 5, init, learningRate);
    TanhLayer tanh3(100, 1);
    FullyConnectedLayer full1(100, 80, init, learningRate);
    TanhLayer tanhFC1(80);
    FullyConnectedLayer full2(80, 10, init, learningRate);
    
    layers.push_back(&conv1);
    layers.push_back(&tanh1);
    layers.push_back(&pool1);
    layers.push_back(&conv2);
    layers.push_back(&tanh2);
    layers.push_back(&pool2);
    layers.push_back(&conv3);
    layers.push_back(&tanh3);
    layers.push_back(&full1);
    layers.push_back(&tanhFC1);
    layers.push_back(&full2);

    ConvolutionNeuralNetwork cnn(layers, sc, minstInput);

    convLayers.push_back(&conv1);
    convLayers.push_back(&conv2);
    convLayers.push_back(&conv3);
    convLayers.push_back(&full1);
    convLayers.push_back(&full2);

    WeightRecorder wr(convLayers, "../MNIST");


    /********************************
     * Supervisors for training
     *********************************
    Validator trainVal(cnn, minstInput, "../MNIST/train", 10);
    Validator valVal(cnn, validateMnist, "../MNIST/validate", 10);
    ActivationVariance avar(layers, "../MNIST/");
    GradientVariance gvar(layers, "../MNIST/");

    cnn.registerSupervisor(&avar);
    cnn.registerSupervisor(&gvar);
    cnn.registerSupervisor(&trainVal);
    cnn.registerSupervisor(&valVal);
    cnn.registerSupervisor(&wr);
    ********************************************/
    
    //train for 10 epochs
    cnn.train(10);
    //for recording weights
    wr.monitor(0);
}


int main(int argc, char *argv[])
{
    trainMnist();
    return 0;
}
