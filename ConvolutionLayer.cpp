#include "layer.hpp"
#include <cblas.h>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include "ConvolutionLayer.hpp"


ConvolutionLayer::ConvolutionLayer(int mapSize, int inputFM,  int outputFM, int kernelSize, Initializer &init, float learningRate) :
        mapSize(mapSize), inputFM(inputFM), outputFM(outputFM), kernelSize(kernelSize), learningRate(learningRate),
        inputMapSize(mapSize + kernelSize - 1),
        bias(vd(outputFM, 0)), 
        output(vvd(outputFM, vd(mapSize*mapSize))),
        prevError(vvd(inputFM, vd(inputMapSize*inputMapSize, 0))),
        kernelW(vvvd(outputFM, vvd(inputFM, vd(kernelSize*kernelSize))))
{
    for (int o = 0; o < outputFM; ++o)
    {
        for (int i = 0; i < inputFM; ++i)
        {
            init.init(kernelW.at(o).at(i));
        }
    }
}    

vvd& ConvolutionLayer::forwardPass(const vvd &input) 
{
    assert(input.size() == inputFM);
    assert(input.at(0).size() == inputMapSize * inputMapSize);

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
    for (int i = 0; i < inputFM; ++i)
    {
        std::fill(prevError.at(i).begin(), prevError.at(i).end(), 0);
    }
    return output;
}

double ConvolutionLayer::convolve(int row, int col, const vvd &input, int outFM)
{
    int inputSize = sqrt(input.at(0).size()); 
    assert(inputSize == mapSize + kernelSize - 1);

    double result = 0;
    for (int i = 0; i < inputFM; ++i)
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

vvd& ConvolutionLayer::backPropagate(const vvd &error) 
{
    assert(error.size() == outputFM);
    assert(error.at(0).size() == mapSize * mapSize);

    for (int ofm = 0; ofm < outputFM; ++ofm)
    {
        for (int oRow = 0; oRow < mapSize; ++oRow)
        {
            for (int oCol = 0; oCol < mapSize; ++oCol)
            {
                for (int ifm = 0; ifm < inputFM; ++ifm)
                {
                    for (int iRow = oRow; iRow < oRow + kernelSize; ++iRow)
                    {
                        cblas_saxpy(kernelSize, error.at(ofm).at(oRow*mapSize + oCol),
                                    (float*) &kernelW.at(ofm).at(ifm).at((iRow-oRow)*kernelSize), 1,
                                    (float*) &prevError.at(ifm).at(iRow*inputMapSize + oCol), 1);
                    }
                }
            }
        }
    }
    update(error);
    return prevError;
}

void ConvolutionLayer::update(const vvd &error)
{
    for (int ofm = 0; ofm < outputFM; ++ofm)
    {
        for (int ifm = 0; ifm < inputFM; ++ifm)
        {
            for (int kRow = 0; kRow < kernelSize; ++kRow)
            {
                for (int kCol = 0; kCol < kernelSize; ++kCol)
                {
                    for (int inRow = kRow; inRow < kRow+mapSize; ++inRow)
                    {
                            //promjene ako cemo probavati mini batch
                            kernelW.at(ofm).at(ifm).at(kRow*kernelSize + kCol) -= learningRate * cblas_dsdot(
                                        mapSize, 
                                        (float*) &(input->at(ifm).at(inRow*inputMapSize+kCol)), 1,
                                        (float*) &error.at(ofm).at((inRow-kRow)*mapSize), 1
                            );
                            std::cout << kernelW.at(ofm).at(ifm).at(kRow*kernelSize + kCol) << std::endl;
                    }
                }
            }
        }
        //promjena biasa
        bias.at(ofm) -= learningRate * std::accumulate(error.at(ofm).begin(), error.at(ofm).end(), 0);
        //std::cout << bias.at(ofm) << std::endl;
    }
}

void ConvolutionLayer::printKernel()
{
    for (int o = 0; o < outputFM; ++o)
    {
        for (int i = 0; i < inputFM; ++i)
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
