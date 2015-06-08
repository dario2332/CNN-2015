#include "layer.hpp"
#include <cblas.h>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ConvolutionLayer.hpp"


ConvolutionLayer::ConvolutionLayer(int mapSize, int prevFM,  int numFM, int kernelSize, Initializer &init, float learningRate) :
        Layer(mapSize + kernelSize - 1, mapSize, prevFM, numFM),  
        kernelSize(kernelSize), learningRate(learningRate),
        bias(vf(numFM, 0)), 
        kernelW(vvvf(numFM, vvf(prevFM, vf(kernelSize*kernelSize))))
{
    for (int o = 0; o < numFM; ++o)
    {
        for (int i = 0; i < prevFM; ++i)
        {
            init.init(kernelW.at(o).at(i), prevFM*prevMapSize*prevMapSize, numFM * mapSize * mapSize);
        }
    }
}    

vvf& ConvolutionLayer::forwardPass(const vvf &input) 
{
    assert(input.size() == prevFM);
    assert(input.at(0).size() == prevMapSize * prevMapSize);

    this->input = &input;
    for (int fm = 0, o = output.size(); fm < o; ++fm)
    {
        for (int row = 0; row < mapSize; ++row)
        {
            for (int col = 0; col < mapSize; ++col)
            {
                output.at(fm).at(row*mapSize + col) = convolve(row, col, input, fm);
                output.at(fm).at(row*mapSize + col) += bias.at(fm);
            }
        }
    }
    
    //reset prevError to 0
    for (int i = 0; i < prevFM; ++i)
    {
        std::fill(prevError.at(i).begin(), prevError.at(i).end(), 0);
    }
    return output;
}

double ConvolutionLayer::convolve(int row, int col, const vvf &input, int outFM)
{
    int inputSize = sqrt(input.at(0).size()); 
    assert(inputSize == mapSize + kernelSize - 1);

    double result = 0;
    for (int i = 0; i < prevFM; ++i)
    {
        for (int j = row; j < row+kernelSize; ++j)
        {
            // moze biti krivo
            result += cblas_dsdot(kernelSize,
                                  (float*)&input.at(i).at(j*inputSize+col), 1, 
                                  (float*)&kernelW.at(outFM).at(i).at((j-row)*kernelSize), 1);
        }
    }
    return result;
}

vvf& ConvolutionLayer::backPropagate(const vvf &error) 
{
    assert(error.size() == numFM);
    assert(error.at(0).size() == mapSize * mapSize);

    for (int ofm = 0; ofm < numFM; ++ofm)
    {
        for (int oRow = 0; oRow < mapSize; ++oRow)
        {
            for (int ifm = 0; ifm < prevFM; ++ifm)
            {
                for (int iRow = oRow; iRow < oRow + kernelSize; ++iRow)
                {
                    for (int oCol = 0; oCol < mapSize; ++oCol)
                    {
                        cblas_saxpy(kernelSize, error.at(ofm).at(oRow*mapSize + oCol),
                                    (float*) &kernelW.at(ofm).at(ifm).at((iRow-oRow)*kernelSize), 1,
                                    (float*) &prevError.at(ifm).at(iRow*prevMapSize + oCol), 1);
                    }
                }
            }
        }
    }
    
    update(error);
    return prevError;
}

void ConvolutionLayer::update(const vvf &error)
{
    for (int ofm = 0; ofm < numFM; ++ofm)
    {
        for (int ifm = 0; ifm < prevFM; ++ifm)
        {
            for (int kRow = 0; kRow < kernelSize; ++kRow)
            {
                for (int kCol = 0; kCol < kernelSize; ++kCol)
                {
                    for (int inRow = kRow; inRow < kRow+mapSize; ++inRow)
                    {
                            float update = learningRate * cblas_sdot(
                                        mapSize, 
                                        (float*) &(input->at(ifm).at(inRow*prevMapSize+kCol)), 1,
                                        (float*) &error.at(ofm).at((inRow-kRow)*mapSize), 1
                            );
                            
                            kernelW.at(ofm).at(ifm).at(kRow*kernelSize + kCol) -= update;
                    }
                }
            }
        }

        bias.at(ofm) -= learningRate * std::accumulate(error.at(ofm).begin(), error.at(ofm).end(), 0);
    }
}

void ConvolutionLayer::printKernel()
{
    for (int o = 0; o < numFM; ++o)
    {
        for (int i = 0; i < prevFM; ++i)
        {
            for (int kRow = 0; kRow < kernelSize; ++kRow)
            {
                for (int kCol = 0; kCol < kernelSize; ++kCol)
                {
                    std::cout << kernelW.at(o).at(i).at(kRow*kernelSize+kCol) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void ConvolutionLayer::loadWeights(std::string file)
{
    std::ifstream in(file, std::fstream::binary);
    int oFM, iFM, kernelSize;
    int outMapSize;
    
    in.read((char*) &oFM, sizeof(int));
    in.read((char*) &iFM, sizeof(int));
    in.read((char*) &kernelSize, sizeof(int));
    in.read((char*) &outMapSize, sizeof(int));
    
    assert(oFM == numFM);
    assert(iFM == prevFM);
    assert(kernelSize == this->kernelSize);
    assert(outMapSize == mapSize);

    for (oFM = 0; oFM < numFM; ++oFM)
    {
        for (iFM = 0; iFM < prevFM; ++iFM)
        {
            for (int k = 0; k < kernelW.at(0).at(0).size(); ++k)
            {
                in.read((char*) &kernelW.at(oFM).at(iFM).at(k), sizeof(float));
            }
        }
    }
    for (int i = 0; i < bias.size(); ++i)
    {
        in.read((char*) &bias.at(i), sizeof(float));
    }
    in.close();
}

void ConvolutionLayer::writeKernel(std::string path)
{
    cv::Mat image(kernelSize, kernelSize, CV_8UC1);
    for (int ofm = 0; ofm < numFM; ++ofm)
    {
        for (int ifm = 0; ifm < prevFM; ++ifm)
        {
            for (int i = 0; i < kernelSize * kernelSize; ++i)
            {
                image.at<unsigned char>(i/kernelSize, i%kernelSize) = kernelW.at(ofm).at(ifm).at(i);
            }
            cv::imwrite(path + "Kernel" + std::to_string(ofm) + "_" + std::to_string(ifm) + ".jpg", image);
        }
    }
}
