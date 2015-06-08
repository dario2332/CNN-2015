#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <random>
#include <string>
#include "Util.hpp"
#include "ConvolutionLayer.hpp"
#include <opencv2/opencv.hpp>

void ReLUInitializer::init(vf &weights, int n_in, int n_out) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(0,2.0/n);
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = distrib(gen);
    }
}


void TestInitializer::init(vf &weights, int n_prev, int n_curr) const
{
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = 1;
    }
}

void TanhInitializer::init(vf &weights, int n_in, int n_out) const
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

vvf& SquareCost::calculate(const vvf &output, const vf& expectedOutput)
{
    assert(output.size() == numOutputs);
    assert(expectedOutput.size() == numOutputs);
    
    error = 0;
    for (int i = 0; i < numOutputs; ++i)
    {
        //std::cout << expectedOutput.at(i) << std::endl;
        assert(output.at(i).size() == 1);
        prevError.at(i).at(0) = output.at(i).at(0) - expectedOutput.at(i);
        error += 0.5 * std::pow(output.at(i).at(0) - expectedOutput.at(i), 2);
    }
    return prevError;
}


MnistSmallInputManager::MnistSmallInputManager(std::string path) : MnistInputManager(20)
{
    expectedOutputs = vvf(20, vf(2));
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
            cv::Mat image(32, 32, CV_8UC1);
            for (int j = 0; j < 32; ++j)
            {
                for (int k = 0; k < 32; ++k)
                {
                    int x = 0;
                    if (j > 1 && j < 30 && k > 1 && k < 30)
                    {
                        inImages.read((char*)&x, sizeof(char));
                    }
                    image.at<unsigned char>(j, k) = x;
                    inputs.at(zeros+ones).at(0).at(j*32+k) = x;
                }
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
    preprocess();
}

MnistInputManager::MnistInputManager (int num, std::string path) : InputManager(num), inputs(vvvf(num, vvf(1, vf(32*32)))),
                                        expectedOutputs(vvf(num, vf(10, 0))), indexes(std::vector<int>(num))
{
    for (int i = 0; i < num; ++i)
    {
        indexes.at(i) = i;
    }
}

void MnistInputManager::preprocess()
{
    for (int image = 0; image < inputs.size(); ++image)
    {
        float sum = 0, mean, variance = 0;
        for (int i = 0; i < 32; ++i)
        {
            for (int j = 0; j < 32; ++j)
            {
                if (j > 1 && j < 30 && i > 1 && i < 30)
                {
                    sum += inputs.at(image).at(0).at(i*32+j);
                } 
            }
        }
        mean = sum / (28*28);
        for (int i = 0; i < 32; ++i)
        {
            for (int j = 0; j < 32; ++j)
            {
                if (j > 1 && j < 30 && i > 1 && i < 30)
                {
                    float x = (inputs.at(image).at(0).at(i*32+j)-mean);
                    variance += x*x;
                } 
            }
        }
        variance /= (28*28);
        for (int i = 0; i < 32; ++i)
        {
            for (int j = 0; j < 32; ++j)
            {
                if (j > 1 && j < 30 && i > 1 && i < 30)
                {
                    inputs.at(image).at(0).at(i*32+j) -= mean;
                    inputs.at(image).at(0).at(i*32+j) /= std::sqrt(variance);
                } 
            }
        }
    }
}

MnistTrainInputManager::MnistTrainInputManager(std::string path) : MnistInputManager(50000) 
{
    std::ifstream inImages, inLabels;
    inImages.open(path + "/train-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    inLabels.open(path + "/train-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(inLabels.is_open());
    assert(inImages.is_open());

    inImages.ignore(4*sizeof(int));
    inLabels.ignore(2*sizeof(int));
    
    for (int i = 0; i < numOfInputs; ++i)
    {
        int label = 0;
        inLabels.read((char*)&label, sizeof(char));
        //static int a = 0;
        //cv::Mat image(32, 32, CV_8UC1);
        for (int j = 0; j < 32; ++j)
        {
            for (int k = 0; k < 32; ++k)
            {
                int x = 0;
                if (j > 1 && j < 30 && k > 1 && k < 30)
                {
                    inImages.read((char*)&x, sizeof(char));

                }
                //image.at<unsigned char>(j, k) = x;
                inputs.at(i).at(0).at(j*32+k) = x;
            }
        }
        expectedOutputs.at(i).at(label) = 1;
        //if (a < 10) cv::imwrite(path + "/image" + std::to_string(a++) + ".jpg", image);
        //else exit(1);
    }
    inImages.close();
    inLabels.close();
    preprocess();
}

MnistValidateInputManager::MnistValidateInputManager(std::string path) : MnistInputManager(10000) 
{
    std::ifstream inImages, inLabels;
    inImages.open(path + "/train-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    inLabels.open(path + "/train-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(inLabels.is_open());
    assert(inImages.is_open());

    inImages.ignore(4*sizeof(int));
    inLabels.ignore(2*sizeof(int));
    inLabels.ignore(50000*sizeof(char));
    inImages.ignore(28*28*50000*sizeof(char));
    
    for (int i = 0; i < numOfInputs; ++i)
    {
        int label = 0;
        inLabels.read((char*)&label, sizeof(char));
        //static int a = 0;
        //cv::Mat image(32, 32, CV_8UC1);
        for (int j = 0; j < 32; ++j)
        {
            for (int k = 0; k < 32; ++k)
            {
                int x = 0;
                if (j > 1 && j < 30 && k > 1 && k < 30)
                {
                    inImages.read((char*)&x, sizeof(char));

                }
                //image.at<unsigned char>(j, k) = x;
                inputs.at(i).at(0).at(j*32+k) = x;
            }
        }
        expectedOutputs.at(i).at(label) = 1;
        //if (a < 10) cv::imwrite(path + "/image" + std::to_string(a++) + ".jpg", image);
        //else exit(1);
    }
    inImages.close();
    inLabels.close();
    preprocess();
}

MnistTestInputManager::MnistTestInputManager(std::string path) : MnistInputManager(10000) 
{
    std::ifstream inImages, inLabels;
    inImages.open(path + "/t10k-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    inLabels.open(path + "/t10k-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(inLabels.is_open());
    assert(inImages.is_open());

    inImages.ignore(4*sizeof(int));
    inLabels.ignore(2*sizeof(int));
    
    for (int i = 0; i < numOfInputs; ++i)
    {
        int label = 0;
        inLabels.read((char*)&label, sizeof(char));
        //static int a = 0;
        //cv::Mat image(32, 32, CV_8UC1);
        for (int j = 0; j < 32; ++j)
        {
            for (int k = 0; k < 32; ++k)
            {
                int x = 0;
                if (j > 1 && j < 30 && k > 1 && k < 30)
                {
                    inImages.read((char*)&x, sizeof(char));

                }
                //image.at<unsigned char>(j, k) = x;
                inputs.at(i).at(0).at(j*32+k) = x;
            }
        }
        expectedOutputs.at(i).at(label) = 1;
        //if (a < 10) cv::imwrite(path + "/image" + std::to_string(a++) + ".jpg", image);
        //else exit(1);
    }
    inImages.close();
    inLabels.close();
    preprocess();
}

void WeightRecorder::monitor(int epoch)
{
    std::string file = path + "/Weights_E" + std::to_string(epoch);
    
    for (int i = 0; i < layers.size(); ++i)
    {
        std::ofstream out(file + "_CL" + std::to_string(i+1), std::fstream::binary | std::fstream::out | std::fstream::trunc);
        vvvf &kernel = layers.at(i) -> getKernel();
        vf &bias = layers.at(i) -> getBias();
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
        for (int i = 0; i < bias.size(); ++i)
        {
            out.write((char*) &bias.at(i), sizeof(float));
        }
        out.close();
    }
}

Validator::Validator (ConvolutionNeuralNetwork &cnn, InputManager& im, std::string path, int numClasses) : 
               TrainingSupervisor(path), cnn(cnn), im(im), possibleOutputs(vvf(numClasses, vf(numClasses, 0)))
{
    for (int i = 0; i < numClasses; ++i)
    {
        possibleOutputs.at(i).at(i) = 1;
    }

}

void Validator::monitor(int epoch)
{
    float error = 0;
    int correct = 0;
    std::vector<std::vector<int> > confusionMatrix(10, std::vector<int>(10, 0));
    
    for (int i = 0, n = im.getInputNum(); i < n; ++i)
    {
        cnn.feedForward(im.getInput(i));
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
        int expected = 0;
        for (int j = 0; j < im.getExpectedOutput(i).size(); ++j)
        {
            if (im.getExpectedOutput(i).at(j) != 0) {expected = j; break;}
        }
        confusionMatrix.at(expected).at(result)++;

    }
    error /= im.getInputNum();
    std::ofstream os(path + "cost", std::ofstream::app); 
    std::ofstream cm(path + "confusionmatrix");

    for (int i = 0; i < confusionMatrix.size(); ++i)
    {
        for (int j = 0; j < confusionMatrix.at(0).size(); ++j)
        {
            cm << confusionMatrix.at(i).at(j) << " ";
        }
        cm << std::endl;
    }
    os << epoch << " " << error << " " << correct << "/" << im.getInputNum() << std::endl;
    std::cout << epoch << " " << error << " " << correct << "/" << im.getInputNum() << std::endl;
    os.close();
    cm.close();

}


void ActivationVariance::monitor(int epoch)
{
    std::ofstream os(path + "ActivationVariance", std::ofstream::app); 
    os << "Epoch: " << epoch << std::endl;
    for (int i = 0; i < layers.size(); ++i)
    {
        float mean = 0;
        float variance = 0;
        vvf& output = layers.at(i)->getOutput();

        for (int j = 0; j < output.size(); ++j)
        {
            for (int k = 0; k < output.at(0).size(); ++k)
            {
                mean += output.at(j).at(k);
            }
        }
        int n = output.size() * output.at(0).size();
        mean /= n;
        for (int j = 0; j < output.size(); ++j)
        {
            for (int k = 0; k < output.at(0).size(); ++k)
            {
                variance += (output.at(j).at(k) - mean) * (output.at(j).at(k) - mean);
            }
        }
        variance /= n;
        
        os << "Layer: " << i << " Variance: " << variance << std::endl;
    }
    os.close();
}

void GradientVariance::monitor(int epoch)
{
    std::ofstream os(path + "GradientVariance", std::ofstream::app); 
    os << "Epoch: " << epoch << std::endl;
    for (int i = 0; i < layers.size(); ++i)
    {
        float mean = 0;
        float variance = 0;
        vvf& error = layers.at(i)->getPrevError();

        for (int j = 0; j < error.size(); ++j)
        {
            for (int k = 0; k < error.at(0).size(); ++k)
            {
                mean += error.at(j).at(k);
            }
        }
        int n = error.size() * error.at(0).size();
        mean /= n;
        for (int j = 0; j < error.size(); ++j)
        {
            for (int k = 0; k < error.at(0).size(); ++k)
            {
                variance += (error.at(j).at(k) - mean) * (error.at(j).at(k) - mean);
            }
        }
        variance /= n;
        
        os << "Layer: " << i << " Variance: " << variance << std::endl;
    }
    os.close();
}
