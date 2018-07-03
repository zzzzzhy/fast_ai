#include <chrono>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfParser/ITfParser.hpp"

namespace py = pybind11;
using namespace std::chrono;

class Pose {
    public:
        Pose() { };
        ~Pose() { };

        void init(std::string model_path);
        int detect(py::array_t<float> image_data,
                 py::array_t<float> heatmap,
                 py::array_t<float> offset_2,
                 py::array_t<float> displacement_fwd_2,
                 py::array_t<float> displacement_bwd_2
		);
        void loop_test(void);

        std::string go(int n_times) {
            std::string result;
            for (int i=0; i<num; ++i)
                result += "woof! ";
            num ++;
            return result;
        };

        int num = 0;
        armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParserPtr(nullptr, nullptr);
        armnn::INetworkPtr network = armnn::INetworkPtr(nullptr, nullptr);
        armnn::IRuntimePtr runtime = armnn::IRuntimePtr(nullptr, nullptr);
	armnn::NetworkId netId;
};

// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

void Pose::init(std::string model_path) {
    this->parser = armnnTfParser::ITfParser::Create();
    char *path = const_cast<char*>(model_path.c_str());
    this->network = parser->CreateNetworkFromBinaryFile(path, 
                                                        { {"image", {1, 513, 513, 3}}},
                                                          { "heatmap",
                                                            "offset_2",
                                                            "displacement_fwd_2",
                                                            "displacement_bwd_2" });
 

    this->runtime = armnn::IRuntime::Create(armnn::Compute::GpuAcc);
    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc, CpuRef
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*(this->network), runtime->GetDeviceSpec());
    // Load the optimized network onto the runtime device
    runtime->LoadNetwork(this->netId, std::move(optNet));
}

int Pose::detect(py::array_t<float> image,
		 py::array_t<float> heatmap,
		 py::array_t<float> offset_2,
		 py::array_t<float> displacement_fwd_2,
		 py::array_t<float> displacement_bwd_2
		)
{
    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = this->parser->GetNetworkInputBindingInfo("image");

    py::buffer_info image_info = image.request();
    auto image_ptr = static_cast<float *>(image_info.ptr);

    py::buffer_info offset_info = offset_2.request();
    auto offset_ptr = static_cast<float *>(offset_info.ptr);

    py::buffer_info fwd_info = displacement_fwd_2.request();
    auto fwd_ptr = static_cast<float *>(fwd_info.ptr);

    py::buffer_info bwd_info = displacement_bwd_2.request();
    auto bwd_ptr = static_cast<float *>(bwd_info.ptr);

    py::buffer_info info = heatmap.request();
    auto heatmap_ptr = static_cast<float *>(info.ptr);
    //std::cout << heatmap_vec.data()[0] << std::endl;
    armnn::OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 0), heatmap_ptr)},
        {1,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 1), offset_ptr)},
        {2,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 2), fwd_ptr)},
        {3,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 3), bwd_ptr)}
    };

    std::cout << "Just Enqueue WorkLoad" << " Image[0]" << image_ptr[0]<< std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
      // armnn::Status ret = runtime->EnqueueWorkload(this->netId,
    runtime->EnqueueWorkload(this->netId,
                             MakeInputTensors(inputBindingInfo, image_ptr),
                             outputTensors);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    for(int i; i < 10; i++){
        std::cout << heatmap_ptr[i] << " ";
    }
    //std::cout << heatmap_vec.data()[0] << std::endl;
    std::cout << "Run time " << duration << std::endl;
    // Convert 1-hot output to an integer label and print
    // long int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    // std::cout << "Predicted: " << label << std::endl;
    // std::cout << "   Actual: " << input->label << std::endl;
    return 0;
}

void Pose::loop_test(void)
{
    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = this->parser->GetNetworkInputBindingInfo("image");
    // Run a single inference on the test image

    std::vector<float> heatmap(1*33*33*17);
    std::vector<float> offset_2(1*33*33*34);
    std::vector<float> displacement_fwd_2(1*33*33*32);
    std::vector<float> displacement_bwd_2(1*33*33*32);

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc, CpuRef
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*(this->network), runtime->GetDeviceSpec());

    // Load the optimized network onto the runtime device
    runtime->LoadNetwork(this->netId, std::move(optNet));

    armnn::OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 0), heatmap.data())},
        {1,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 1), offset_2.data())},
        {2,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 2), displacement_fwd_2.data())},
        {3,armnn::Tensor(runtime->GetOutputTensorInfo(this->netId, 3), displacement_bwd_2.data())}
    };

    std::array<float, 1*513*513*3> image_data;
    std::cout << "Just Enqueue WorkLoad" << std::endl;

    for(int i = 0; i < 100; i++){
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      // armnn::Status ret = runtime->EnqueueWorkload(this->netId,
      runtime->EnqueueWorkload(this->netId,
                               MakeInputTensors(inputBindingInfo, &image_data[0]),
                               outputTensors);
      high_resolution_clock::time_point t2 = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>( t2 - t1 ).count();

      std::cout << "Run time " << duration << std::endl;
    }
    // Convert 1-hot output to an integer label and print
    // long int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    // std::cout << "Predicted: " << label << std::endl;
    // std::cout << "   Actual: " << input->label << std::endl;
}

PYBIND11_MODULE(pose, m) {
    py::class_<Pose>(m, "Pose")
      .def(py::init<>())
      .def("go", &Pose::go)
      .def("loop_test", &Pose::loop_test)
      .def("init", &Pose::init)
      .def("detect", &Pose::detect);
}
