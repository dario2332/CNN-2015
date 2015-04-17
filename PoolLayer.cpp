#include "PoolLayer.hpp"
#include <cassert>

PoolLayer::PoolLayer(int kernelSize, int numFM, int inputMapSize) : kernelSize(kernelSize), 
              numFM(numFM), inMapSize(inputMapSize),
              outMapSize(inputMapSize/kernelSize),  
              output(vvd(numFM, vd(outMapSize*outMapSize))),
              prevError(vvd(numFM, vd(inMapSize*inMapSize, 0)))
{
    assert(inMapSize % kernelSize == 0);
}

vvd& MaxPoolLayer::forwardPass(const vvd &input)
{
    //reset prevError
    for (int i = 0; i < numFM; ++i)
    {
        std::fill(prevError.at(i).begin(), prevError.at(i).end(), 0);
    }

    assert(input.at(0).size() == inMapSize*inMapSize);
    assert(input.size() == numFM);
    
    this->input = &input;
    
    for (int fm = 0; fm < numFM; ++fm)
    {
        for (int oRow = 0; oRow < outMapSize; oRow++)
        {
            for (int oCol = 0; oCol < outMapSize; oCol++)
            {
                output.at(fm).at(oRow*outMapSize+oCol) = max(fm, oRow*kernelSize, oCol*kernelSize);
            }
        }
    }
    return output;
}


float MaxPoolLayer::max(int fm, int iRow, int iCol)
{
    float max = input->at(fm).at(iRow*inMapSize+iCol);
    int row = iRow, col = iCol;
    for (int i = iRow; i < iRow+kernelSize; ++i)
    {
        for (int j = iCol; j < iCol+kernelSize; ++j)
        {
            if (input->at(fm).at(i*inMapSize + j) > max)
            {
                max = input->at(fm).at(i*inMapSize + j);
                row = i; 
                col = j;
            }
        }
    }
    prevError.at(fm).at(row*inMapSize+col) = 1;
    return max;
}

vvd& MaxPoolLayer::backPropagate(const vvd &error)
{
    assert(error.size() == numFM);
    assert(error.at(0).size() == outMapSize*outMapSize);

    for (int fm = 0; fm < numFM; ++fm)
    {
        for (int iRow = 0; iRow < inMapSize; iRow++)
        {
            for (int iCol = 0; iCol < inMapSize; iCol++)
            {
                prevError.at(fm).at(iRow*inMapSize+iCol) *= error.at(fm).at(iRow/kernelSize * outMapSize + iCol/kernelSize);
            }
        }
    }
    return prevError;
}
