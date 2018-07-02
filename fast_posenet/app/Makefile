ARMNN_LIB = ${ARMNN_DIR}/build
ARMNN_INC = ${ARMNN_DIR}/include

LIB_BOOST = -lboost_system-mt -lboost_filesystem -lboost_thread -lboost_serialization -lboost_date_time -lboost_log -lboost_program_options
LIBS = $(LIB_BOOST) -lpthread -lprotobuf -static

all: mnist_tf

mnist_caffe: mnist_caffe.cpp mnist_loader.hpp
	g++ -O3 -std=c++14 -I$(ARMNN_INC) mnist_caffe.cpp -o mnist_caffe -L$(ARMNN_LIB) -larmnn -larmnnCaffeParser $(LIBS)

mnist_tf: mnist_tf.cpp mnist_loader.hpp
	g++ -O3 -I$(ARMNN_INC) mnist_tf.cpp -o mnist_tf -L$(ARMNN_LIB) -larmnn -larmnnTfParser $(LIBS)

clean:
	-rm -f mnist_tf mnist_caffe

test: mnist_caffe mnist_tf
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_caffe
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_tf
