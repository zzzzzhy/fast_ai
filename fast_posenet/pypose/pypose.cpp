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

        void init(void);
        int detect(void);

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
};

// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

void Pose::init(void) {
    this->parser = armnnTfParser::ITfParser::Create();
    this->network = parser->CreateNetworkFromBinaryFile("model/posenet.pb",
                                                                   { {"image", {1, 513, 513, 3}}},
                                                                   { "heatmap",
                                                                     "offset_2",
                                                                     "displacement_fwd_2",
                                                                     "displacement_bwd_2" });
}

int Pose::detect(void)
{
    armnn::IRuntimePtr runtime=armnn::IRuntime::Create(armnn::Compute::GpuAcc);

    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = this->parser->GetNetworkInputBindingInfo("image");
    // Run a single inference on the test image

    std::vector<float> heatmap(1*33*33*384);
    std::vector<float> offset_2(1*33*33*34);
    std::vector<float> displacement_fwd_2(1*33*33*32);
    std::vector<float> displacement_bwd_2(1*33*33*32);

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc, CpuRef
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*(this->network), runtime->GetDeviceSpec());

    // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    armnn::OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), heatmap.data())},
        {1,armnn::Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 1), offset_2.data())},
        {2,armnn::Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 2), displacement_fwd_2.data())},
        {3,armnn::Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 3), displacement_bwd_2.data())}
    };

    std::array<float, 1*513*513*3> image_data;
    std::cout << "Just Enqueue WorkLoad" << std::endl;

    for(int i = 0; i < 100; i++){
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      // armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
      runtime->EnqueueWorkload(networkIdentifier,
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
    return 0;
}

PYBIND11_MODULE(pose, m) {
    py::class_<Pose>(m, "Pose")
      .def(py::init<>())
      .def("go", &Pose::go)
      .def("init", &Pose::init)
      .def("detect", &Pose::detect);
}
