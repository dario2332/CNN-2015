#include "util.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <opencv2/opencv.hpp>

namespace cnn {

void ReLUInitializer::init(int n_in, int n_out, vf &weights) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> distrib(0,2.0/n_);
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = distrib(gen);
    }
}


void TestInitializer::init(int n_in, int n_out, vf &weights) const
{
    for (int i = 0; i < weights.size(); ++i)
    {
        weights.at(i) = 1;
    }
}

void TanhInitializer::init(int n_in, int n_out, vf &weights) const
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

vvf& SquareCost::calculate(const vvf &output, const vf& expected_output)
{
    assert(output.size() == num_outputs_);
    assert(expected_output.size() == num_outputs_);
    
    error_ = 0;
    for (int i = 0; i < num_outputs_; ++i)
    {
        //std::cout << expected_output.at(i) << std::endl;
        assert(output.at(i).size() == 1);
        prev_error_.at(i).at(0) = output.at(i).at(0) - expected_output.at(i);
        error_ += 0.5 * std::pow(output.at(i).at(0) - expected_output.at(i), 2);
    }
    return prev_error_;
}

// This class was only build for testing purposes
MnistSmallInputManager::MnistSmallInputManager(std::string path) : MnistInputManager(20)
{
    expected_outputs_ = vvf(20, vf(2));
    std::ifstream in_images, in_labels;
    in_images.open(path + "/train-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    in_labels.open(path + "/train-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(in_labels.is_open());
    assert(in_images.is_open());

    in_images.ignore(4*sizeof(int));
    in_labels.ignore(2*sizeof(int));

    int zeros = 0, ones = 0;
    while (zeros < 10 || ones < 10)
    {
        int label = 0;
        in_labels.read((char*)&label, sizeof(char));
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
                        in_images.read((char*)&x, sizeof(char));
                    }
                    image.at<unsigned char>(j, k) = x;
                    inputs_.at(zeros+ones).at(0).at(j*32+k) = x;
                }
            }
            if (label == 0)
            {
                expected_outputs_.at(zeros+ones) = {1, 0};
                zeros++;
            }
            else 
            {
                expected_outputs_.at(zeros+ones) = {0, 1};
                ones++;
            }
            cv::imwrite(path + "/image" + std::to_string(a++) + ".jpg", image);
        }
        else 
        {
            in_images.ignore(28*28*sizeof(char));
        }
        label = 0;
    }
    in_images.close();
    in_labels.close();
    preprocess();
}

MnistInputManager::MnistInputManager (int num, std::string path) : InputManager(num), inputs_(vvvf(num, vvf(1, vf(32*32)))),
                                        expected_outputs_(vvf(num, vf(10, 0))), indexes_(std::vector<int>(num))
{
    for (int i = 0; i < num; ++i)
    {
        indexes_.at(i) = i;
    }
}

void MnistInputManager::readData(std::ifstream &in_images, std::ifstream &in_labels)
{
    for (int i = 0; i < num_inputs_; ++i)
    {
        int label = 0;
        in_labels.read((char*)&label, sizeof(char));
        //static int a = 0;
        //cv::Mat image(32, 32, CV_8UC1);
        for (int j = 0; j < 32; ++j)
        {
            for (int k = 0; k < 32; ++k)
            {
                int x = 0;
                if (j > 1 && j < 30 && k > 1 && k < 30)
                {
                    in_images.read((char*)&x, sizeof(char));

                }
                //image.at<unsigned char>(j, k) = x;
                inputs_.at(i).at(0).at(j*32+k) = x;
            }
        }
        expected_outputs_.at(i).at(label) = 1;
        //if (a < 10) cv::imwrite(path + "/image" + std::to_string(a++) + ".jpg", image);
        //else exit(1);
    }
    in_images.close();
    in_labels.close();
    preprocess();
}

void MnistInputManager::preprocess()
{
    for (int image = 0; image < inputs_.size(); ++image)
    {
        float sum = 0, mean, variance = 0;
        for (int i = 0; i < 32; ++i)
        {
            for (int j = 0; j < 32; ++j)
            {
                if (j > 1 && j < 30 && i > 1 && i < 30)
                {
                    sum += inputs_.at(image).at(0).at(i*32+j);
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
                    float x = (inputs_.at(image).at(0).at(i*32+j)-mean);
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
                    inputs_.at(image).at(0).at(i*32+j) -= mean;
                    inputs_.at(image).at(0).at(i*32+j) /= std::sqrt(variance);
                } 
            }
        }
    }
}

MnistTrainInputManager::MnistTrainInputManager(std::string path) : MnistInputManager(50000) 
{
    std::ifstream in_images, in_labels;
    in_images.open(path + "/train-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    in_labels.open(path + "/train-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(in_labels.is_open());
    assert(in_images.is_open());

    in_images.ignore(4*sizeof(int));
    in_labels.ignore(2*sizeof(int));

    readData(in_images, in_labels);
}

MnistValidateInputManager::MnistValidateInputManager(std::string path) : MnistInputManager(10000) 
{
    std::ifstream in_images, in_labels;
    in_images.open(path + "/train-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    in_labels.open(path + "/train-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(in_labels.is_open());
    assert(in_images.is_open());

    in_images.ignore(4*sizeof(int));
    in_labels.ignore(2*sizeof(int));
    in_labels.ignore(50000*sizeof(char));
    in_images.ignore(28*28*50000*sizeof(char));
    
    readData(in_images, in_labels);
}

MnistTestInputManager::MnistTestInputManager(std::string path) : MnistInputManager(10000) 
{
    std::ifstream in_images, in_labels;
    in_images.open(path + "/t10k-images.idx3-ubyte", std::fstream::binary | std::fstream::in);
    in_labels.open(path + "/t10k-labels.idx1-ubyte", std::fstream::binary | std::fstream::in);

    assert(in_labels.is_open());
    assert(in_images.is_open());

    in_images.ignore(4*sizeof(int));
    in_labels.ignore(2*sizeof(int));
    
    readData(in_images, in_labels);
}

void WeightRecorder::monitor(int epoch)
{
    std::string file = path_ + "/Weights_E" + std::to_string(epoch);
    
    for (int i = 0; i < layers_.size(); ++i)
    {
        std::ofstream out(file + "_CL" + std::to_string(i+1), std::fstream::binary | std::fstream::out | std::fstream::trunc);
        const vvvf &kernel = layers_.at(i) -> getKernel();
        const vf &bias = layers_.at(i) -> getBias();
        int num_out_feature_maps = kernel.size(), num_in_feature_maps = kernel.at(0).size();
        int kernel_size = std::sqrt(kernel.at(0).at(0).size());
        int out_map_size = layers_.at(i) -> getMapSize();
        
        out.write((char*) &num_out_feature_maps, sizeof(int));
        out.write((char*) &num_in_feature_maps, sizeof(int));
        out.write((char*) &kernel_size, sizeof(int));
        out.write((char*) &out_map_size, sizeof(int));

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

Validator::Validator (ConvolutionNeuralNetwork &cnn, InputManager& im, std::string path, int num_classes) 
    : TrainingSupervisor(path), cnn_(cnn), input_manager_(im), 
      possible_outputs_(vvf(num_classes, vf(num_classes, 0)))
{
    for (int i = 0; i < num_classes; ++i)
    {
        possible_outputs_.at(i).at(i) = 1;
    }
}

void Validator::monitor(int epoch)
{
    float error = 0;
    int correct = 0;
    std::vector<std::vector<int> > confusion_matrix(10, std::vector<int>(10, 0));
    std::vector<int> rank(10, 0);
    
    for (int i = 0, n = input_manager_.getInputNum(); i < n; ++i)
    {
        cnn_.feedForward(input_manager_.getInput(i));
        error += cnn_.getCost(input_manager_.getExpectedOutput(i));
        float min = cnn_.getCost(possible_outputs_.at(0));
        int result = 0;
        std::vector<float> costs;
        costs.push_back(min);
        
        for (int j = 1; j < possible_outputs_.size(); ++j)
        {
            float x = cnn_.getCost(possible_outputs_.at(j));
            costs.push_back(x);
            if (x < min)
            {
                min = x;
                result = j;
            }
        }

        // Rank
        float correct_cost;
        for (int j = 0; j < input_manager_.getExpectedOutput(i).size(); ++j)
        {
            if (input_manager_.getExpectedOutput(i).at(j) == 1) correct_cost = costs.at(j);
        }
        std::sort(costs.begin(), costs.end());
        rank.at(std::find(costs.begin(), costs.end(), correct_cost) - costs.begin())++;
        
        // Cost
        if (possible_outputs_.at(result) == input_manager_.getExpectedOutput(i))
        { 
            correct ++;
        }

        // Confusion matrix
        int expected = 0;
        for (int j = 0; j < input_manager_.getExpectedOutput(i).size(); ++j)
        {
            if (input_manager_.getExpectedOutput(i).at(j) != 0) {expected = j; break;}
        }
        confusion_matrix.at(expected).at(result)++;

    }
    error /= input_manager_.getInputNum();
    std::ofstream os(path_ + "cost", std::ofstream::app); 
    std::ofstream cm(path_ + "confusionmatrix");
    std::ofstream score(path_ + "rank");
    for (int i = 0; i < rank.size(); ++i)
    {
        score << rank.at(i) << " ";
    }

    for (int i = 0; i < confusion_matrix.size(); ++i)
    {
        for (int j = 0; j < confusion_matrix.at(0).size(); ++j)
        {
            cm << confusion_matrix.at(i).at(j) << " ";
        }
        cm << std::endl;
    }
    os << epoch << " " << error << " " << correct << "/" << input_manager_.getInputNum() << std::endl;
    std::cout << epoch << " " << error << " " << correct << "/" << input_manager_.getInputNum() << std::endl;
    os.close();
    cm.close();
    score.close();
}


void ActivationVariance::monitor(int epoch)
{
    std::ofstream os(path_ + "ActivationVariance", std::ofstream::app); 
    os << "Epoch: " << epoch << std::endl;
    for (int i = 0; i < layers_.size(); ++i)
    {
        float mean = 0;
        float variance = 0;
        const vvf& output = layers_.at(i)->getOutput();

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
    std::ofstream os(path_ + "GradientVariance", std::ofstream::app); 
    os << "Epoch: " << epoch << std::endl;
    for (int i = 0; i < layers_.size(); ++i)
    {
        float mean = 0;
        float variance = 0;
        const vvf& error = layers_.at(i)->getPrevError();

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

} // namespace cnn

