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

void testMnist()
{
    std::vector<Layer*> layers;
    MnistTrainInputManager minstInput;
    MnistValidateInputManager validateMnist;
    MnistTestInputManager testMnist;
    SquareCost sc(10);
    const float learningRate = 0.001;

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
    
    Validator test(cnn, testMnist, "../MNIST/test", 10);

    conv1.loadWeights("../MNIST/Weights_E0_CL1");
    conv2.loadWeights("../MNIST/Weights_E0_CL2");
    conv3.loadWeights("../MNIST/Weights_E0_CL3");
    full1.loadWeights("../MNIST/Weights_E0_CL4");
    full2.loadWeights("../MNIST/Weights_E0_CL5");
    test.monitor(0);

}

int main(int argc, char *argv[])
{
    testMnist();
    return 0;
}
