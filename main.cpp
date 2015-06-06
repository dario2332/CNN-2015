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

void testForwardPass()
{
    TestInitializer init;
    ConvolutionLayer l(28, 3, 10, 5, init, 1);
    vvf input(3, vf(32*32, 1));
    l.forwardPass(input);
    
    for (int i = 0; i < l.getOutput().size(); ++i)
    {
        for (int j = 0; j < l.getOutput().at(0).size(); ++j)
        {
            double d = l.getOutput().at(i).at(j); 
            assert(d == 75);
        }
    }
}

void testBackPropagation()
{
    TestInitializer init;
    ConvolutionLayer l(6, 3, 1, 3, init, 1);
    vvf input(3, vf(8*8, 1));
    l.forwardPass(input);
    vvf error(1, vf(6*6, 2));
    l.backPropagate(error);
    
    
    for (int i = 0; i < l.getPrevError().size(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                double d = l.getPrevError().at(i).at(j*8+k); 
                std::cout << d << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    //-35 treba biti sve
    l.printKernel();
    
}

void testForwardPassTime(int n)
{
    // za n = 100 48x48 trajalo je 78s
    TestInitializer init;
    ConvolutionLayer l(44, 7, 10, 5, init, 1);
    vvf input(7, vf(48*48, 1));
    for (int i = 0; i < n * 10; i ++)
    {
        l.forwardPass(input);
    }

}

void testMaxPool()
{
    MaxPoolLayer mpl(2, 1, 4);
    vvf input(1);
    input.at(0) = { 2, 3, 4, 7,
                    9, 3, 4, 5,
                    11, 2, 1, 0,
                    2, 2, 4, 4
                   };
    mpl.forwardPass(input);
    vf expectedPrevError = {0, 0, 0, 1,
                            1, 0, 0, 0,
                            1, 0, 0, 0,
                            0, 0, 1, 0
                            };
    vf expectedOutput = { 9, 7,
                          11, 4 };
    
    assert(mpl.getPrevError().at(0) == expectedPrevError);
    assert(mpl.getOutput().at(0) == expectedOutput);

    vvf error(1);
    error.at(0) = {2, 3,
                   -1, 0};
    vf expectedPrevError2 ={0, 0, 0, 3,
                            2, 0, 0, 0,
                            -1, 0, 0, 0,
                            0, 0, 0, 0
                            };
    mpl.backPropagate(error);
    assert(mpl.getPrevError().at(0) == expectedPrevError2);
}

void testTanhLayer()
{
    TanhLayer tl(1, 2);
    vvf input(1);
    input.at(0) = { 2, 3,
                    9, 3 };
    tl.forwardPass(input);
    vf expectedOutput = {1.49294, 1.65417, 1.71588, 1.65417};
    vf expectedPrevError = {0.701763, 0.306059, 3.54885e-05, 0.20404};
    for (int i = 0; i < tl.getOutput().size(); ++i)
    {
        for (int j = 0; j < tl.getOutput().at(0).size(); ++j)
        {
            double d = tl.getOutput().at(i).at(j); 
            assert(d - expectedOutput.at(j) < 1e-5);
            //std::cout << d << " ";
        }
    }
    //std::cout << std::endl;

    vvf error(1);
    error.at(0) = { 2, 3,
                    1, 2 };
    tl.backPropagate(error);

    for (int i = 0; i < tl.getPrevError().size(); ++i)
    {
        for (int j = 0; j < tl.getPrevError().at(0).size(); ++j)
        {
            double d = tl.getPrevError().at(i).at(j); 
            assert(d - expectedPrevError.at(j) < 1e-5);
            //std::cout << d << " ";
        }
    }
    //std::cout << std::endl;
}

void testActivationLayerTime(int n)
{
    //for n = 1000000 it took 54 seconds
    vvf input(80, vf(1, 1));
    vvf error(80, vf(1, 1));
    TanhLayer tl(80, 1);
    for (int i = 0; i < n; ++i)
    {
        tl.forwardPass(input);
        tl.backPropagate(error);
    }
}

void testSquareCost()
{
    SquareCost sc(3);
    vvf output(3);
    output.at(0).push_back(1);
    output.at(1).push_back(2);
    output.at(2).push_back(3);
    vf expectedOutput = {2, 2.5, 2};
    vvf expectedPrevError(3);
    expectedPrevError.at(0).push_back(-1);
    expectedPrevError.at(1).push_back(-0.5);
    expectedPrevError.at(2).push_back(1);
    sc.calculate(output, expectedOutput);
    assert(sc.getError() == 1.125);
    assert(sc.getPrevError() == expectedPrevError);
}

void testCNN()
{
    std::vector<Layer*> layers;
    std::vector<ConvolutionLayer*> convLayers;
    MnistSmallInputManager minstInput;
    SquareCost sc(2);
    const float learningRate = 0.002;

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
    FullyConnectedLayer full2(80, 2, init, learningRate);
    //TanhLayer tanhFC2(2);
    
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
    //layers.push_back(&tanhFC2);

    ConvolutionNeuralNetwork cnn(layers, sc, minstInput);
    
    convLayers.push_back(&conv1);
    convLayers.push_back(&conv2);
    convLayers.push_back(&conv3);
    convLayers.push_back(&full1);
    convLayers.push_back(&full2);

    WeightRecorder wr(convLayers, "MNIST");
    Validator val(cnn, minstInput, "MNIST/train", 2);
    ActivationVariance avar(layers, "MNIST/");
    GradientVariance gvar(layers, "MNIST/");

    cnn.registerSupervisor(&avar);
    cnn.registerSupervisor(&gvar);
    cnn.registerSupervisor(&val);
    cnn.registerSupervisor(&wr);

    cnn.train(100);

    conv1.loadWeights("MNIST/Weights_E19_CL1");
    conv2.loadWeights("MNIST/Weights_E19_CL2");
    conv3.loadWeights("MNIST/Weights_E19_CL3");
    full1.loadWeights("MNIST/Weights_E19_CL4");
    full2.loadWeights("MNIST/Weights_E19_CL5");
    cnn.train(2);
    conv1.writeKernel("MNIST/");
    
}

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
    //TanhLayer sigmFC2(10);
    
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
    //layers.push_back(&sigmFC2);

    ConvolutionNeuralNetwork cnn(layers, sc, minstInput);
    
    convLayers.push_back(&conv1);
    convLayers.push_back(&conv2);
    convLayers.push_back(&conv3);
    convLayers.push_back(&full1);
    convLayers.push_back(&full2);

    WeightRecorder wr(convLayers, "MNIST");
    Validator trainVal(cnn, minstInput, "MNIST/train", 10);
    Validator valVal(cnn, validateMnist, "MNIST/validate", 10);
    ActivationVariance avar(layers, "MNIST/");
    GradientVariance gvar(layers, "MNIST/");

    cnn.registerSupervisor(&avar);
    cnn.registerSupervisor(&gvar);
    cnn.registerSupervisor(&trainVal);
    cnn.registerSupervisor(&valVal);
    cnn.registerSupervisor(&wr);
    //15.5s je svo potrebno ucitavanje

    //32.05s je 1000 backward passova
    //cnn.forwardPass(minstInput.getInput(0));
    //for (int i = 0; i < 1000; ++i)
    //{
    //    cnn.backPropagate(sc.calculate(layers.at(layers.size()-1)->getOutput(), minstInput.getExpectedOutput(0)));
    //}

    //14.75s je 1000 forward passova
    //for (int i = 0; i < 1000; ++i)
    //{
    //    cnn.forwardPass(minstInput.getInput(0));
    //}
    //trainVal.monitor(0);
    //valVal.monitor(0);
    conv1.loadWeights("MNIST/Weights_E10_CL1");
    conv2.loadWeights("MNIST/Weights_E10_CL2");
    conv3.loadWeights("MNIST/Weights_E10_CL3");
    full1.loadWeights("MNIST/Weights_E10_CL4");
    full2.loadWeights("MNIST/Weights_E10_CL5");

    cnn.train(10);
}

int main(int argc, char *argv[])
{
    //testForwardPass();
    //testBackPropagation();
    //testForwardPassTime(100);
    //testMaxPool();
    //testTanhLayer();
    //testActivationLayerTime(1000000);
    //testSquareCost();
    //testCNN();
    trainMnist();
    return 0;
}
