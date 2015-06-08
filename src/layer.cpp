#include "layer.hpp"


Layer::Layer(int prevMapSize, int mapSize, int prevFM,  int numFM) : 
            prevMapSize(prevMapSize), mapSize(mapSize), 
            prevFM(prevFM), numFM(numFM), output(vvf(numFM, vf(mapSize*mapSize))), 
            prevError(vvf(prevFM, vf(prevMapSize*prevMapSize, 0))) {}

