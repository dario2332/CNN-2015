#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <random>
#include "Util.hpp"
#include "ConvolutionLayer.hpp"

void ReLUInitializer::init(vd &weights) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(0,2.0/n);
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = distrib(gen);
    }
}


void TestInitializer::init(vd &weights) const
{
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = 1;
    }
}

vvd& SquareCost::calculate(const vvd &output, const vd& expectedOutput)
{
    assert(output.size() == numOutputs);
    assert(expectedOutput.size() == numOutputs);
    
    error = 0;
    for (int i = 0; i < numOutputs; ++i)
    {
        assert(output.at(i).size() == 1);
        prevError.at(i).at(0) = output.at(i).at(0) - expectedOutput.at(i);
        error += 0.5 * std::pow(output.at(i).at(0) - expectedOutput.at(i), 2);
    }
    return prevError;
}


MnistTestInputManager::MnistTestInputManager(std::string path) : InputManager(20, 20, 20), inputs(vvvd(20, vvd(1, vd(28*28)))),
                                                                 expectedOutputs(vvd(20, vd(2)))
{
    int zeros = 0, ones = 0;
    std::ifstream inImages, inLabels;
    inImages.open(path + "/train-images-idx3-ubyte.gz", std::ios::binary);
    inLabels.open(path + "/train-labels-idx1-ubyte.gz", std::ios::binary);
    inImages.ignore(4*sizeof(int));
    inLabels.ignore(2*sizeof(int));

    while (zeros < 10 && ones < 10)
    {
        int label = 0;
        inLabels.read((char*)&label, sizeof(char));
        if (label == 0 && zeros < 10 || label == 1 && ones < 10)
        {
            for (int i = 0; i < 28*28; ++i)
            {
                int x = 0;
                inImages.read((char*)&x, sizeof(char));
                inputs.at(zeros+ones).at(0).at(i) = x;
            }
            if (label == 0)
            {
                expectedOutputs.at(zeros+ones) = {1, 0};
                zeros++;
            }
            else 
            {
                expectedOutputs.at(zeros+ones) = {0, 1};
                ones++;
            }
        }
        else 
        {
            inImages.ignore(28*28*sizeof(char));
        }
        label = 0;
    }
    inImages.close();
    inLabels.close();
}


void WeightRecorder::monitor(int epoch)
{
    std::string file = datasetName + "/Weights_E" + std::to_string(epoch);
    
    for (int i = 0; i < layers.size(); ++i)
    {
        std::ofstream out(file + "_CL" + std::to_string(i+1), std::fstream::binary | std::fstream::out | std::fstream::trunc);
        vvvd kernel = layers.at(i) -> getKernel();
        int oFM = kernel.size(), iFM = kernel.at(0).size(), kernelSize = std::sqrt(kernel.at(0).at(0).size());
        float learningRate = layers.at(i) -> getLearningRate();
        int outMapSize = layers.at(i) -> getMapSize();
        
        out.write((char*) &oFM, sizeof(int));
        out.write((char*) &iFM, sizeof(int));
        out.write((char*) &kernelSize, sizeof(int));
        out.write((char*) &learningRate, sizeof(float));
        out.write((char*) &outMapSize, sizeof(int));

        for (int i = 0; i < kernel.size(); ++i)
        {
            for (int j = 0; j < kernel.at(0).size(); ++j)
            {
                for (int k = 0; k < kernel.at(0).at(0).size(); ++k)
                {
                    out.write((char*) &kernel.at(i).at(j).at(k), sizeof(float));
                }
            }
        }
        out.close();
    }
}
