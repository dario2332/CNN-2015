#include "PoolLayer.hpp"
#include <cassert>
#include <iostream>

PoolLayer::PoolLayer(int frameSize, int numFM, int prevMapSize) : Layer(prevMapSize, prevMapSize/frameSize, numFM, numFM),
              frameSize(frameSize)
{
    assert(prevMapSize % frameSize == 0);
}

vvf& MaxPoolLayer::forwardPass(const vvf &input)
{
    //reset prevError
    for (int i = 0; i < numFM; ++i)
    {
        std::fill(prevError.at(i).begin(), prevError.at(i).end(), 0);
    }

    assert(input.at(0).size() == prevMapSize*prevMapSize);
    assert(input.size() == numFM);
    
    this->input = &input;
    
    for (int fm = 0; fm < numFM; ++fm)
    {
        for (int oRow = 0; oRow < mapSize; oRow++)
        {
            for (int oCol = 0; oCol < mapSize; oCol++)
            {
                output.at(fm).at(oRow*mapSize+oCol) = max(fm, oRow*frameSize, oCol*frameSize);
            }
        }
    }
    return output;
}


float MaxPoolLayer::max(int fm, int iRow, int iCol)
{
    float max = input->at(fm).at(iRow*prevMapSize+iCol);
    int row = iRow, col = iCol;
    for (int i = iRow; i < iRow+frameSize; ++i)
    {
        for (int j = iCol; j < iCol+frameSize; ++j)
        {
            if (input->at(fm).at(i*prevMapSize + j) > max)
            {
                max = input->at(fm).at(i*prevMapSize + j);
                row = i; 
                col = j;
            }
        }
    }
    prevError.at(fm).at(row*prevMapSize+col) = 1;
    return max;
}

vvf& MaxPoolLayer::backPropagate(const vvf &error)
{
    assert(error.size() == numFM);
    assert(error.at(0).size() == mapSize*mapSize);

    for (int fm = 0; fm < numFM; ++fm)
    {
        for (int iRow = 0; iRow < prevMapSize; iRow++)
        {
            for (int iCol = 0; iCol < prevMapSize; iCol++)
            {
                prevError.at(fm).at(iRow*prevMapSize+iCol) *= error.at(fm).at(iRow/frameSize * mapSize + iCol/frameSize);
            }
        }
    }
    return prevError;
}
