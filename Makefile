CC = g++
CFLAGS = -O3 -std=c++11

all: build/test build/train 

build/test:
	mkdir -p build
	${CC} ${CFLAGS}  src/test.cc src/convolution_neural_network.cc src/convolution_layer.cc src/activation_layer.cc src/pool_layer.cc src/util.cc src/layer.cc -o build/test -lopenblas `pkg-config opencv --libs`

build/train:
	mkdir -p build
	${CC} ${CFLAGS}  src/train.cc src/convolution_neural_network.cc src/convolution_layer.cc src/activation_layer.cc src/pool_layer.cc src/util.cc src/layer.cc  -o build/train -lopenblas `pkg-config opencv --libs`

clean:
	rm -f build/*
