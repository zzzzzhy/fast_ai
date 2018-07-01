//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfParser/ITfParser.hpp"

#include "mnist_loader.hpp"


// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData)
{
    return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
}

int main(int argc, char** argv)
{
    // Load a test image and its correct label
    std::string dataDir = "data/";
    //int testImageIndex = 0;

    // Import the TensorFlow model. Note: use CreateNetworkFromBinaryFile for .pb files.
    armnnTfParser::ITfParserPtr parser = armnnTfParser::ITfParser::Create();
    // armnn::INetworkPtr network = parser->CreateNetworkFromTextFile("model/simple_mnist_tf.prototxt",
    /*armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("model/freezed.pb",
                                                                   { {"input", {1, 160, 160, 3}}},
                                                                   { "embeddings" });

    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("input");
    //armnnTfParser::BindingPointInfo inputBindingInfo2 = parser->GetNetworkInputBindingInfo("phase_train");
    armnnTfParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("embeddings");
    */
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile("model/posenet.pb",
                                                                   { {"image", {1, 513, 513, 3}}},
                                                                   { "heatmap",
                                                                     "offset_2",
                                                                     "displacement_fwd_2",
                                                                     "displacement_bwd_2" });

    // Find the binding points for the input and output nodes
    armnnTfParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("image");
    //armnnTfParser::BindingPointInfo inputBindingInfo2 = parser->GetNetworkInputBindingInfo("phase_train");
    //armnnTfParser::BindingPointInfo heatmap_info = parser->GetNetworkOutputBindingInfo("heatmap");
    //armnnTfParser::BindingPointInfo offset_2_info = parser->GetNetworkOutputBindingInfo("offset_2");
    //armnnTfParser::BindingPointInfo displacement_fwd_2_info = parser->GetNetworkOutputBindingInfo("displacement_fwd_2");
    //armnnTfParser::BindingPointInfo displacement_bwd_2_info = parser->GetNetworkOutputBindingInfo("displacement_bwd_2");

    // Run a single inference on the test image

    std::vector<float> heatmap(1*33*33*384);
    std::vector<float> offset_2(1*33*33*34);
    std::vector<float> displacement_fwd_2(1*33*33*32);
    std::vector<float> displacement_bwd_2(1*33*33*32);


    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc, CpuRef
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::Compute::CpuRef);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, runtime->GetDeviceSpec());

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
    // armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
    std::cout << "Just Enqueue WorkLoad" << std::endl;
    runtime->EnqueueWorkload(networkIdentifier,
                             MakeInputTensors(inputBindingInfo, &image_data[0]),
                             outputTensors);

    // Convert 1-hot output to an integer label and print
    // long int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    // std::cout << "Predicted: " << label << std::endl;
    // std::cout << "   Actual: " << input->label << std::endl;
    return 0;
}
