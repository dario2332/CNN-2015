#include <iostream>
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
    vvd input(3, vd(32*32, 1));
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
    vvd input(3, vd(8*8, 1));
    l.forwardPass(input);
    vvd error(1, vd(6*6, 2));
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
    vvd input(7, vd(48*48, 1));
    for (int i = 0; i < n * 10; i ++)
    {
        l.forwardPass(input);
    }

}

void testMaxPool()
{
    MaxPoolLayer mpl(2, 1, 4);
    vvd input(1);
    input.at(0) = { 2, 3, 4, 7,
                    9, 3, 4, 5,
                    11, 2, 1, 0,
                    2, 2, 4, 4
                   };
    mpl.forwardPass(input);
    vd expectedPrevError = {0, 0, 0, 1,
                            1, 0, 0, 0,
                            1, 0, 0, 0,
                            0, 0, 1, 0
                            };
    vd expectedOutput = { 9, 7,
                          11, 4 };
    
    assert(mpl.getPrevError().at(0) == expectedPrevError);
    assert(mpl.getOutput().at(0) == expectedOutput);

    vvd error(1);
    error.at(0) = {2, 3,
                   -1, 0};
    vd expectedPrevError2 ={0, 0, 0, 3,
                            2, 0, 0, 0,
                            -1, 0, 0, 0,
                            0, 0, 0, 0
                            };
    mpl.backPropagate(error);
    assert(mpl.getPrevError().at(0) == expectedPrevError2);
}

void testSigmoidLayer()
{
    SigmoidLayer sl(1, 2);
    vvd input(1);
    input.at(0) = { 2, 3,
                    9, 3 };
    sl.forwardPass(input);
    vd expectedOutput = {1.49294, 1.65417, 1.71588, 1.65417};
    vd expectedPrevError = {0.701763, 0.306059, 3.54885e-05, 0.20404};
    for (int i = 0; i < sl.getOutput().size(); ++i)
    {
        for (int j = 0; j < sl.getOutput().at(0).size(); ++j)
        {
            double d = sl.getOutput().at(i).at(j); 
            assert(d - expectedOutput.at(j) < 1e-5);
            //std::cout << d << " ";
        }
    }
    //std::cout << std::endl;

    vvd error(1);
    error.at(0) = { 2, 3,
                    1, 2 };
    sl.backPropagate(error);

    for (int i = 0; i < sl.getPrevError().size(); ++i)
    {
        for (int j = 0; j < sl.getPrevError().at(0).size(); ++j)
        {
            double d = sl.getPrevError().at(i).at(j); 
            assert(d - expectedPrevError.at(j) < 1e-5);
            //std::cout << d << " ";
        }
    }
    //std::cout << std::endl;
}

void testActivationLayerTime(int n)
{
    //for n = 1000000 it took 54 seconds
    vvd input(80, vd(1, 1));
    vvd error(80, vd(1, 1));
    SigmoidLayer sl(80, 1);
    for (int i = 0; i < n; ++i)
    {
        sl.forwardPass(input);
        sl.backPropagate(error);
    }
}

void testSquareCost()
{
    SquareCost sc(3);
    vvd output(3);
    output.at(0).push_back(1);
    output.at(1).push_back(2);
    output.at(2).push_back(3);
    vd expectedOutput = {2, 2.5, 2};
    vvd expectedPrevError(3);
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
    MnistTestInputManager minstInput;
    SquareCost sc(2);

    TestInitializer init;
    
    ConvolutionLayer conv1(24, 1, 8, 5, init, 0.01);
    SigmoidLayer sigm1(8, 24);
    MaxPoolLayer pool1(2, 8, 24);
    ConvolutionLayer conv2(10, 8, 20, 3, init, 0.01);
    SigmoidLayer sigm2(20, 10);
    MaxPoolLayer pool2(2, 20, 10);
    ConvolutionLayer conv3(1, 20, 80, 5, init, 0.01);
    SigmoidLayer sigm3(80, 1);
    FullyConnectedLayer full1(80, 40, init, 0.01);
    SigmoidLayer sigmFC1(40);
    FullyConnectedLayer full2(40, 2, init, 0.01);
    
    layers.push_back(&conv1);
    layers.push_back(&sigm1);
    layers.push_back(&pool1);
    layers.push_back(&conv2);
    layers.push_back(&sigm2);
    layers.push_back(&pool2);
    layers.push_back(&conv3);
    layers.push_back(&sigm3);
    layers.push_back(&full1);
    layers.push_back(&sigmFC1);
    layers.push_back(&full2);

    ConvolutionNeuralNetwork cnn(layers, sc, minstInput);
    
    convLayers.push_back(&conv1);
    convLayers.push_back(&conv2);
    convLayers.push_back(&conv3);
    convLayers.push_back(&full1);
    convLayers.push_back(&full2);

    WeightRecorder wr(convLayers, "MNIST");

    cnn.registerSupervisor(&wr);
    cnn.train(2, true);


}
int main(int argc, char *argv[])
{
    //testForwardPass();
    //testBackPropagation();
    //testForwardPassTime(100);
    //testMaxPool();
    //testSigmoidLayer();
    //testActivationLayerTime(1000000);
    //testSquareCost();
    testCNN();
    return 0;
}
