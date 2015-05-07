#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <random>
#include <string>
#include "Util.hpp"
#include "ConvolutionLayer.hpp"
#include <opencv2/opencv.hpp>

void ReLUInitializer::init(vd &weights, int n_in, int n_out) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(0,2.0/n);
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = distrib(gen);
    }
}


void TestInitializer::init(vd &weights, int n_prev, int n_curr) const
{
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = 1;
    }
}

void SigmoidInitializer::init(vd &weights, int n_in, int n_out) const
{
    float l = -4 * sqrt(6.0 / (n_in + n_out));
    float r = 4 * sqrt(6.0 / (n_in + n_out));
    static std::default_random_engine generator(time(NULL));
    std::uniform_real_distribution<float> distribution(l ,r);

    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = distribution(generator);
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


MnistSmallInputManager::MnistSmallInputManager(std::string path) : InputManager(20), inputs(vvvd(20, vvd(1, vd(28*28)))),
                                                                 expectedOutputs(vvd(20, vd(2)))
{
    int x = 0;
    int zeros = 0, ones = 0;
    std::ifstream inImages, inLabels;
    inImages.open(path + "/train-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    inLabels.open(path + "/train-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(inLabels.is_open());
    assert(inImages.is_open());

    inImages.ignore(4*sizeof(int));
    inLabels.ignore(2*sizeof(int));

    while (zeros < 10 || ones < 10)
    {
        int label = 0;
        inLabels.read((char*)&label, sizeof(char));
        if (label == 0 && zeros < 10 || label == 1 && ones < 10)
        {
            static int a = 0;
            cv::Mat image(28, 28, CV_8UC1);
            for (int i = 0; i < 28*28; ++i)
            {
                int x = 0;
                inImages.read((char*)&x, sizeof(char));
                inputs.at(zeros+ones).at(0).at(i) = x;
                image.at<unsigned char>(i/28, i%28) = x;
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
            cv::imwrite(path + "/image" + std::to_string(a++) + ".jpg", image);
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
    std::string file = path + "/Weights_E" + std::to_string(epoch);
    
    for (int i = 0; i < layers.size(); ++i)
    {
        std::ofstream out(file + "_CL" + std::to_string(i+1), std::fstream::binary | std::fstream::out | std::fstream::trunc);
        vvvd kernel = layers.at(i) -> getKernel();
        int oFM = kernel.size(), iFM = kernel.at(0).size(), kernelSize = std::sqrt(kernel.at(0).at(0).size());
        int outMapSize = layers.at(i) -> getMapSize();
        
        out.write((char*) &oFM, sizeof(int));
        out.write((char*) &iFM, sizeof(int));
        out.write((char*) &kernelSize, sizeof(int));
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

Validator::Validator (ConvolutionNeuralNetwork &cnn, InputManager& im, std::string path, int numClasses) : 
               TrainingSupervisor(path), cnn(cnn), im(im), possibleOutputs(vvd(numClasses, vd(numClasses, 0)))
{
    for (int i = 0; i < numClasses; ++i)
    {
        possibleOutputs.at(i).at(i) = 1;
    }

}

void Validator::monitor(int epoch)
{
    InputManager& im = cnn.getInputManager();
    float error = 0;
    int correct = 0;
    
    for (int i = 0, n = im.getInputNum(); i < n; ++i)
    {
        cnn.forwardPass(im.getInput(i));
        error += cnn.getCost(im.getExpectedOutput(i));
        float min = cnn.getCost(possibleOutputs.at(0));
        int result = 0;
        
        for (int j = 1; j < possibleOutputs.size(); ++j)
        {
            float x = cnn.getCost(possibleOutputs.at(j));
            if (x < min)
            {
                min = x;
                result = j;
            }
        }
        if (possibleOutputs.at(result) == im.getExpectedOutput(i))
        { 
            correct ++;
        }
    }
    error /= im.getInputNum();
    std::ofstream os(path + "cost", std::ofstream::app); 

    os << epoch << " " << error << " " << correct << "/" << im.getInputNum() << std::endl;
    os.close();
}
