CC = g++
CFLAGS = -O3 -std=c++11

all: build/test build/train 

build/test:
	mkdir -p build
	${CC} ${CFLAGS}  src/test.cpp src/CNN.cpp src/ConvolutionLayer.cpp src/ActivationLayer.cpp src/PoolLayer.cpp src/Util.cpp src/layer.cpp -o build/test -lopenblas `pkg-config opencv --libs`

build/train:
	mkdir -p build
	${CC} ${CFLAGS}  src/train.cpp src/CNN.cpp src/ConvolutionLayer.cpp src/ActivationLayer.cpp src/PoolLayer.cpp src/Util.cpp src/layer.cpp -o build/train -lopenblas `pkg-config opencv --libs`

clean:
	rm -f build/*
