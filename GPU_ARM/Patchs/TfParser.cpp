//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "TfParser.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/Utils.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Descriptors.hpp>

#include <GraphTopologicalSort.hpp>
#include <Permute.hpp>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/polymorphic_cast.hpp>

#include <memory>
#include <sstream>
#include <numeric>
#include <functional>

using namespace armnn;

namespace armnnTfParser
{
namespace
{

const PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
const PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };

IConnectableLayer* AddSwizzleLayer(INetwork& network, IOutputSlot& input, const PermutationVector& mapping,
    const std::string& name)
{
    // Add swizzle layer
    IConnectableLayer* const layer = network.AddPermuteLayer(mapping, name.c_str());

    // Connect intput to swizzle layer
    input.Connect(layer->GetInputSlot(0));

    // Setup swizzled output
    const TensorInfo outInfo = armnnUtils::Permuted(input.GetTensorInfo(), mapping);
    layer->GetOutputSlot(0).SetTensorInfo(outInfo);

    return layer;
}

IConnectableLayer* SwizzleInDeswizzleOut(INetwork& network, IOutputSlot& input, IConnectableLayer& layer,
    const std::string& name)
{
    // Add swizzle layer
    IConnectableLayer* const swizzleLayer = AddSwizzleLayer(network, input, NHWCToArmNN, "swizzle_for-" + name);

    // Connect swizzledInput to layer
    swizzleLayer->GetOutputSlot(0).Connect(layer.GetInputSlot(0));

    // Add deswizzle layer
    IConnectableLayer* const deswizzleLayer = AddSwizzleLayer(network, layer.GetOutputSlot(0), ArmNNToNHWC,
        "deswizzle_for-" + name);

    return deswizzleLayer;
}

template <typename Callable>
void ReadMandatoryNodeAttributeImpl(const tensorflow::NodeDef& nodeDef,
    const std::string& attribName,
    tensorflow::AttrValue::ValueCase expectedValueCase,
    Callable callable)
{
    auto iter = nodeDef.attr().find(attribName);
    if (iter != nodeDef.attr().end())
    {
        const auto& attrValue = iter->second;
        if (attrValue.value_case() == expectedValueCase)
        {
            callable(attrValue);
        }
        else
        {
            throw ParseException(boost::str(boost::format(
                "Attribute %1% of node %2% expected to have %3% as tensorflow::AttrValue::ValueCase, "
                "but found %4% instead")
                % attribName
                % nodeDef.name()
                % static_cast<int>(expectedValueCase)
                % static_cast<int>(attrValue.value_case())));
        }
    }
    else
    {
        throw ParseException(boost::str(boost::format("Could not find required attribute %1% in node %2%")
            % attribName % nodeDef.name()));
    }
}

template <typename Callable>
void ReadOptionalNodeAttributeImpl(const tensorflow::NodeDef& nodeDef,
    const std::string& attribName,
    tensorflow::AttrValue::ValueCase expectedValueCase,
    Callable callable)
{
    auto iter = nodeDef.attr().find(attribName);
    if (iter != nodeDef.attr().end())
    {
        const auto& attrValue = iter->second;
        if (attrValue.value_case() == expectedValueCase)
        {
            callable(attrValue);
        }
        else
        {
            throw ParseException(boost::str(boost::format(
                "Attribute %1% of node %2% expected to have %3% as tensorflow::AttrValue::ValueCase, "
                "but found %4% instead")
                % attribName
                % nodeDef.name()
                % static_cast<int>(expectedValueCase)
                % static_cast<int>(attrValue.value_case())));
        }
    }
}

float ReadMandatoryNodeFloatAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    float attribValue = 0.0f;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kF,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.f();
    });
    return attribValue;
}

uint32_t ReadMandatoryNodeUint32Attribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    uint32_t attribValue = 0u;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kI,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = static_cast<uint32_t>(attrValue.i());
    });
    return attribValue;
}

std::string ReadMandatoryNodeStringAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    std::string attribValue = "";
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kS,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.s();
    });
    return attribValue;
}

std::vector<uint32_t> ReadMandatoryNodeUint32ListAttribute(const tensorflow::NodeDef& nodeDef,
    const std::string& name)
{
    std::vector<uint32_t> attriList;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kList,
        [&attriList](const tensorflow::AttrValue& attrValue)
    {
        for (int attriNum = 0; attriNum < attrValue.list().i_size(); ++attriNum)
        {
            attriList.push_back(static_cast<uint32_t>(attrValue.list().i(attriNum)));
        }
    });

    return attriList;
}

std::vector<uint32_t> ReadOptionalNodeUint32ListAttribute(const tensorflow::NodeDef& nodeDef,
    const std::string& name)
{
    std::vector<uint32_t> attriList;
    ReadOptionalNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kList,
        [&attriList](const tensorflow::AttrValue& attrValue)
    {
        for (int attriNum = 0; attriNum < attrValue.list().i_size(); ++attriNum)
        {
            attriList.push_back(static_cast<uint32_t>(attrValue.list().i(attriNum)));
        }
    });

    return attriList;
}

bool ReadOptionalNodeBoolAttribute(const tensorflow::NodeDef& nodeDef,
    const std::string& name,
    bool defaultValue = false)
{
    bool attribValue = defaultValue;
    ReadOptionalNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kB,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.b();
    });
    return attribValue;
}

tensorflow::DataType ReadMandatoryNodeTypeAttribute(const tensorflow::NodeDef& nodeDef, const std::string& name)
{
    tensorflow::DataType attribValue = tensorflow::DT_INVALID;
    ReadMandatoryNodeAttributeImpl(nodeDef, name, tensorflow::AttrValue::kType,
        [&attribValue](const tensorflow::AttrValue& attrValue)
    {
        attribValue = attrValue.type();
    });
    return attribValue;
}

TensorInfo PrepareReshape(const TensorInfo& input, const std::vector<int32_t>& targetDims)
{
    std::vector<unsigned int> outDims(targetDims.begin(), targetDims.end());
    const auto stretchDim = std::find(targetDims.begin(), targetDims.end(), -1);

    if (stretchDim != targetDims.end())
    {
        if (std::find(std::next(stretchDim), targetDims.end(), -1) != targetDims.end())
        {
            throw ParseException("At most one component of shape can be -1");
        }

        auto targetNumElements = boost::numeric_cast<unsigned int>(std::accumulate(targetDims.begin(), targetDims.end(),
            -1, std::multiplies<int32_t>()));
        auto stretchIndex = static_cast<size_t>(std::distance(targetDims.begin(), stretchDim));
        outDims[stretchIndex] = input.GetNumElements() / targetNumElements;
    }

    TensorInfo reshapeInfo = input;
    reshapeInfo.SetShape(TensorShape{ static_cast<unsigned int>(outDims.size()), outDims.data() });

    return reshapeInfo;
}

// We need the input0Slot to guide the reshape for input1Slot
IOutputSlot* BroadcastForAddandMul(IOutputSlot* input0Slot, IOutputSlot* input1Slot, bool isNHWC, INetwork& m_Network,
                                   const tensorflow::NodeDef& nodeDef)
{
    const TensorInfo& input1Info = input1Slot->GetTensorInfo();
    const TensorInfo inputTensorInfo = input0Slot->GetTensorInfo();
    const unsigned int matchDim = inputTensorInfo.GetNumDimensions() - (isNHWC ? 1 : 3);
    std::array<unsigned int, MaxNumOfTensorDimensions> reshapedDimensions;
    std::fill_n(reshapedDimensions.begin(), inputTensorInfo.GetNumDimensions(), 1);
    reshapedDimensions[matchDim] = input1Info.GetShape()[0];

    armnn::TensorInfo reshapedInfo = input1Info;
    reshapedInfo.SetShape(TensorShape{ inputTensorInfo.GetNumDimensions(), reshapedDimensions.data() });

    const std::string reshapeLayerName = "reshape_for-" + nodeDef.name();
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = reshapedInfo.GetShape();
    IConnectableLayer* const reshapeLayer = m_Network.AddReshapeLayer(reshapeDesc, reshapeLayerName.c_str());

    input1Slot->Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);

    input1Slot = &reshapeLayer->GetOutputSlot(0);

    return input1Slot;
}

OutputId ParseOutputId(const std::string & name)
{
    unsigned int outputNum = 0;
    size_t colonPos = name.find_last_of(":");
    if (colonPos != std::string::npos)
    {
        int n = std::stoi(name.substr(colonPos+1));
        if (n<0 || n>100)
        {
            throw ParseException("Output tensor id is out of range for "+name);
        }
        outputNum = static_cast<unsigned int>(n);
    }
    return OutputId(name.substr(0,colonPos),outputNum);
}

} // namespace

const std::map<std::string, TfParser::OperationParsingFunction> TfParser::ms_OperationNameToParsingFunctions = {
    { "Const",                 &TfParser::ParseConst },
    { "Add",                   &TfParser::ParseAdd },
    { "BiasAdd",               &TfParser::ParseBiasAdd },
    { "Identity",              &TfParser::ParseIdentity },
    { "Conv2D",                &TfParser::ParseConv2D },
    { "DepthwiseConv2dNative", &TfParser::ParseDepthwiseConv2D },
    { "FusedBatchNorm",        &TfParser::ParseFusedBatchNorm },
    { "ConcatV2",              &TfParser::ParseConcat },
    { "LRN",                   &TfParser::ParseLrn },
    { "MatMul",                &TfParser::ParseMatMul },
    { "Mul",                   &TfParser::ParseMul },
    { "Placeholder",           &TfParser::ParsePlaceholder },
    { "Relu",                  &TfParser::ParseRelu },
    { "Relu6",                 &TfParser::ParseRelu6 },
    { "Reshape",               &TfParser::ParseReshape },
    { "ResizeBilinear",        &TfParser::ParseResizeBilinear },
    { "Shape",                 &TfParser::ParseShape },
    { "Squeeze",               &TfParser::ParseSqueeze },
    { "Sigmoid",               &TfParser::ParseSigmoid },
    { "Softmax",               &TfParser::ParseSoftmax },
    { "Softplus",              &TfParser::ParseSoftplus },
    { "Tanh",                  &TfParser::ParseTanh },
    { "MaxPool",               &TfParser::ParseMaxPool },
    { "AvgPool",               &TfParser::ParseAvgPool },
};

ITfParser* ITfParser::CreateRaw()
{
    return new TfParser();
}

ITfParserPtr ITfParser::Create()
{
    return ITfParserPtr(CreateRaw(), &ITfParser::Destroy);
}

void ITfParser::Destroy(ITfParser* parser)
{
    delete parser;
}

inline void CalculateSamePadding(uint32_t inputSize, uint32_t stride,
                                 uint32_t filterSize, bool samePadding,
                                 uint32_t* paddingFront, uint32_t* paddingBack) {
    *paddingFront = 0;
    *paddingBack = 0;

    if (samePadding) {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t temp = (outputSize - 1) * stride + filterSize;
        if (temp > inputSize) {
            *paddingFront = (temp - inputSize) / 2;
            *paddingBack = (temp - inputSize) - *paddingFront;
        }
    }
}

void CalcPadding(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t& outPadHead, uint32_t& outPadTail,
                 bool samePadding)
{
    CalculateSamePadding(input, stride, kernel, samePadding, &outPadHead, &outPadTail);
}

/// An Abstract base class which represents a single tensorflow operation (node)
/// that has been (potentially partially) converted to Armnn.
/// It may not yet have been fully converted into actual Armnn layers.
class ParsedTfOperation
{
public:
    ParsedTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
    : m_Parser(parser)
    , m_Node(node)
    {
    }

    virtual ~ParsedTfOperation() {};

    const tensorflow::NodeDef& GetNode() const { return m_Node; }

    /// Gets the ArmNN IOutputSlot corresponding to the given output index of the Tensorflow operation.
    /// This may result in the creation of Armnn layers if this was deferred (e.g. see ParsedConstTfOperation).
    virtual IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) = 0;

    /// If this operation is an Identity then this will follow return the 'parent' operation (recursively).
    virtual ParsedTfOperation* ResolveIdentityOperations()
    {
        return this;
    }

protected:
    TfParser* m_Parser;
    const tensorflow::NodeDef& m_Node;
};

/// An ParsedTfOperation where the Armnn equivalent is a single layer,
/// with output slots that correspond directly to the Tf node outputs.
class SingleLayerParsedTfOperation : public ParsedTfOperation
{
public:
    SingleLayerParsedTfOperation(TfParser* parser, const tensorflow::NodeDef& node, IConnectableLayer* layer)
    : ParsedTfOperation(parser, node)
    , m_Layer(layer)
    {
    }

    IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) override
    {
        BOOST_ASSERT(m_Layer);
        // Assume one-to-one mapping between Tf and armnn output slots.
        unsigned int armnnOutputSlotIdx = tfOutputIndex;
        if (armnnOutputSlotIdx >= m_Layer->GetNumOutputSlots())
        {
            throw ParseException(
                boost::str(boost::format("The requested output slot #%1% "
                    "for %2% does not exist") % armnnOutputSlotIdx % m_Layer->GetName()));
        }
        return m_Layer->GetOutputSlot(armnnOutputSlotIdx);
    }

protected:
    IConnectableLayer* m_Layer;
};

/// A SingleLayerParsedTfOperation for deferred layer creation
class DeferredSingleLayerParsedTfOperation : public SingleLayerParsedTfOperation
{
public:
    DeferredSingleLayerParsedTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
    : SingleLayerParsedTfOperation(parser, node, nullptr)
    {
    }

    IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) override
    {
        if (!m_Layer)
        {
            CreateLayerDeferred();
        }
        return SingleLayerParsedTfOperation::ResolveArmnnOutputSlot(tfOutputIndex);
    }

private:
    virtual void CreateLayerDeferred() = 0;
};


TfParser::TfParser()
    : m_Network(nullptr, nullptr)
{
}


const tensorflow::NodeDef* TfParser::ResolveIdentityNode(const tensorflow::NodeDef* nodeDef)
{
    if (nodeDef->op() != "Identity")
    {
        return nodeDef;
    }

    if (nodeDef->input_size() != 1)
    {
        throw ParseException("Identity node does not have correct amount of inputs!");
    }

    auto it = m_NodesByName.find(nodeDef->input(0));
    if (it != m_NodesByName.end())
    {
        const tensorflow::NodeDef* inputNode = it->second;
        return ResolveIdentityNode(inputNode);
    }
    else
    {
        throw ParseException("Cannot find what the Identity node is linked to!");
    }
}

std::vector<OutputOfConstNodeDef>
TfParser::GetTfInputNodes(const tensorflow::NodeDef& nodeDef) const
{
    std::vector<OutputOfConstNodeDef> ret;

    if (nodeDef.op() == "Const")
    {
        // For some reason const node can have "Control Inputs". We ignore them for now.
        return ret;
    }

    ret.reserve(boost::numeric_cast<size_t>(nodeDef.input_size()));
    for (int j = 0; j < nodeDef.input_size(); ++j)
    {
        OutputId outputId = ParseOutputId(nodeDef.input(j));

        /*if (nodeDef.input(j)[0] == '^') // I couldn't find a better test for control inputs.
        {
            throw ParseException(
                "Node '" + nodeDef.name() + "' has Control Input '" + nodeDef.input(j) + "' which is unsupported.");
        }*/

        auto inputIt = m_NodesByName.find(outputId.m_IndexedValue);
        if (inputIt == m_NodesByName.end())
        {
            throw ParseException(
                "Can't find node '" + nodeDef.input(j) +
                "', which is listed as an input of '" + nodeDef.name() + "'");
        }
        ret.push_back(OutputOfConstNodeDef(inputIt->second,outputId.m_Index));
    }

    return ret;
}

std::vector<OutputOfParsedTfOperation>
TfParser::GetInputParsedTfOperationsChecked(const tensorflow::NodeDef& nodeDef,
                                            std::size_t expectedNumInputs)
{
    // Fetch the tensorflow nodes connected as inputs and validate the size.
    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);
    const std::size_t numInputs = nodes.size();
    if (numInputs != expectedNumInputs)
    {
        throw ParseException(boost::str(boost::format("Unexpected number of inputs for node %1%. "
            "Expected %2%, found %3%") % nodeDef.name() % expectedNumInputs % numInputs));
    }
    // Fetch the corresponding ParsedTfOperation operations
    std::vector<OutputOfParsedTfOperation> result;
    for (auto&& node : nodes)
    {
        auto it = m_ParsedTfOperations.find(node.m_IndexedValue->name());
        if (it == m_ParsedTfOperations.end())
        {
            throw ParseException("Node with name '" + node.m_IndexedValue->name() + "' has not been parsed");
        }
        ParsedTfOperation* parsedOp = it->second.get();
        // Transparently 'skip' any Identity operations. This simplifies the logic inside the ParseXXX() functions.
        parsedOp = parsedOp->ResolveIdentityOperations();
        result.push_back(OutputOfParsedTfOperation(parsedOp,node.m_Index));
    }
    return result;
}

ParsedTfOperationPtr TfParser::ParseAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    // If one of the inputs is a MatMul and the other is a const, then we handle both nodes together as FullyConnected
    if (inputs[0].m_IndexedValue->GetNode().op() == "MatMul" &&
        HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        IConnectableLayer* layer =
            AddFullyConnectedLayer(inputs[0].m_IndexedValue->GetNode(),
                                   &nodeDef,nodeDef.name().c_str());
        return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
    }
    else if (HasParsedConstTensor<float>(inputs[0].m_IndexedValue->GetNode().name()) &&
                                         inputs[1].m_IndexedValue->GetNode().op() == "MatMul")
    {
        IConnectableLayer* layer =
            AddFullyConnectedLayer(inputs[1].m_IndexedValue->GetNode(),
                                   &nodeDef,nodeDef.name().c_str());
        return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
    }
    else
    {
        // Otherwise it's just a regular addition
        return AddAdditionLayer(nodeDef);
    }
}

ParsedTfOperationPtr TfParser::ParseBiasAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    return AddAdditionLayer(nodeDef, true);
}

/// An ParsedTfOperation which forwards to another (used for Identity nodes).
class ParsedIdentityTfOperation : public ParsedTfOperation
{
public:
    ParsedIdentityTfOperation(TfParser* parser, const tensorflow::NodeDef& node, ParsedTfOperation* representative)
        : ParsedTfOperation(parser, node)
        , m_Representative(representative)
    {
    }

    virtual IOutputSlot& ResolveArmnnOutputSlot(unsigned int tfOutputIndex) override
    {
        BOOST_ASSERT(m_Representative);
        return m_Representative->ResolveArmnnOutputSlot(tfOutputIndex);
    }

    virtual ParsedTfOperation* ResolveIdentityOperations() override
    {
        return m_Representative->ResolveIdentityOperations();
    }

private:
    ParsedTfOperation* m_Representative;
};

ParsedTfOperationPtr TfParser::ParseIdentity(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
    // Any requests for the output slots of this node should be forwarded to the node connected as input.
    return std::make_unique<ParsedIdentityTfOperation>(this, nodeDef, inputs[0].m_IndexedValue);
}

/// An ParsedTfOperation for a Const node.
/// Creation of the armnn ConstLayer is deferred until it is actually needed, because Const nodes are mostly used
/// for weight inputs to MatMul/Conv2D nodes and in these cases armnn doesn't need a ConstLayer.
template <typename T>
class ParsedConstTfOperation : public DeferredSingleLayerParsedTfOperation
{
public:
    ParsedConstTfOperation(TfParser* parser, const tensorflow::NodeDef& node,
        const T* tensorData, const TensorInfo& tensorInfo)
        : DeferredSingleLayerParsedTfOperation(parser, node),
        m_Storage(tensorData, tensorData + tensorInfo.GetNumElements()),
        m_TensorInfo(tensorInfo)
    {
        BOOST_ASSERT(tensorInfo.GetDataType() == GetDataType<T>());
    }

    void CreateLayerDeferred() override
    {
        BOOST_ASSERT(m_Layer == nullptr);
        m_Layer = m_Parser->m_Network->AddConstantLayer(ConstTensor(m_TensorInfo, m_Storage), m_Node.name().c_str());
        m_Layer->GetOutputSlot(0).SetTensorInfo(m_TensorInfo);
    }

    ConstTensor GetConstTensor(bool swizzleForConvolutionWeights, std::vector<T>& outputTensorData) const
    {
        // Mappings from TensorFlow filter tensors to the ArmNN filter tensors.
        // Tensorflow weights are [H, W, In, Out]
        // ArmNN weights are [Out, In, H, W]
        static const PermutationVector HWIOToOIHW = {2, 3, 1, 0};

        const TensorInfo outInfo = swizzleForConvolutionWeights
                                   ? armnnUtils::Permuted(m_TensorInfo, HWIOToOIHW)
                                   : m_TensorInfo;

        outputTensorData.resize(m_TensorInfo.GetNumElements());

        // Copy or swizzle from the permanent storage into the storage the caller provided.
        if (swizzleForConvolutionWeights)
        {
            armnnUtils::Permute(outInfo.GetShape(), HWIOToOIHW, m_Storage.data(), outputTensorData.data());
        }
        else
        {
            memcpy(outputTensorData.data(), m_Storage.data(), m_TensorInfo.GetNumBytes());
        }
        // Update the result to point to the user provided storage
        ConstTensor constTensor(outInfo, outputTensorData);
        return constTensor;
    }

private:
    ///< Manages the lifetime of the tensor data.
    std::vector<T> m_Storage;
    ///< Describes the layout of the tensor and points to the data in m_Storage.
    TensorInfo m_TensorInfo;
};

DataType ConvertTfTensorDataType(const tensorflow::DataType tfDataType)
{
    switch (tfDataType)
    {
    case tensorflow::DT_FLOAT:
        return DataType::Float32;
        break;
    case tensorflow::DT_INT32:
        return DataType::Signed32;
        break;
    default:
        throw ParseException(boost::str(
            boost::format("Unknown DataType %1% for node")
            % tensorflow::DataType_Name(tfDataType)));
    }
}

struct ParseTfTensorValueList
{
    template<typename DataType>
    static void Parse(
        const tensorflow::TensorProto& tfTensor,
        unsigned int dstElements,
        std::vector<int8_t>& outputData);

    template <typename DataType>
    static void ReadData(const void* srcData, unsigned int numSrcElements,
        std::vector<int8_t>& dstData, unsigned int numDstElements)
    {
        // If there are no entries in the list, perform no action
        if (numSrcElements == 0)
        {
            return;
        }

        // If no size was provided, use the length of the value list
        if (numDstElements == 0)
        {
            numDstElements = numSrcElements;
        }

        // Allocate memory
        dstData.resize(std::max(numSrcElements, numDstElements) * sizeof(DataType));

        const DataType* srcTensor = reinterpret_cast<const DataType*>(srcData);
        DataType* dstTensor = reinterpret_cast<DataType*>(dstData.data());

        // Copy the value list entries into the destination
        std::copy(srcTensor, srcTensor + numSrcElements, dstTensor);

        if (numDstElements > numSrcElements)
        {
            // Use the last element in the list to fill the remaining entries
            std::fill(dstTensor + numSrcElements, dstTensor + numDstElements, srcTensor[numSrcElements - 1]);
        }
    }

};

template <>
void ParseTfTensorValueList::Parse<float>(const tensorflow::TensorProto& tfTensor,
    unsigned int dstElements, std::vector<int8_t>& outputData)
{
    ReadData<float>(tfTensor.float_val().data(), static_cast<unsigned int>(tfTensor.float_val_size()),
        outputData, dstElements);
}

template <>
void ParseTfTensorValueList::Parse<int32_t>(const tensorflow::TensorProto& tfTensor,
    unsigned int dstElements, std::vector<int8_t>& outputData)
{
    ReadData<int32_t>(tfTensor.int_val().data(), static_cast<unsigned int>(tfTensor.int_val_size()),
        outputData, dstElements);
}

template <template<typename> class OperatorType, typename T = int8_t>
struct MakeTfOperation
{
    template<typename DataType, class... Args>
    inline static std::unique_ptr<OperatorType<DataType>> Parse(TfParser* parser, const tensorflow::NodeDef& node,
        Args&&... args)
    {
        return std::make_unique<OperatorType<DataType>>(parser, node, std::forward<Args>(args)...);
    }
};

template <>
struct MakeTfOperation<ParsedConstTfOperation>
{
    template<typename DataType, class... Args>
    inline static std::unique_ptr<ParsedConstTfOperation<DataType>> Parse(TfParser* parser,
        const tensorflow::NodeDef& node, const std::vector<int8_t>& tensorData, const TensorInfo& tensorInfo)
    {
        return std::make_unique<ParsedConstTfOperation<DataType>>(parser, node,
            reinterpret_cast<const DataType*>(tensorData.data()), tensorInfo);
    }
};

template <class FuncType>
struct InvokeParseFunction
{
    template<class ResType, class... Args>
    inline static ResType Result(DataType dataType, Args&&... args)
    {
        if (dataType == DataType::Float32)
        {
            return FuncType::template Parse<float>(std::forward<Args>(args)...);
        }
        else if (dataType == DataType::Signed32)
        {
            return FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
        }

        return ResType();
    }

    template<class... Args>
    inline static void Result(DataType dataType, Args&&... args)
    {
        if (dataType == DataType::Float32)
        {
            FuncType::template Parse<float>(std::forward<Args>(args)...);
        }
        else if (dataType == DataType::Signed32)
        {
            FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
        }
    }
};

ParsedTfOperationPtr TfParser::ParseConst(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    BOOST_ASSERT(nodeDef.op() == "Const");

    if (nodeDef.attr().count("value") == 0)
    {
        throw ParseException(boost::str(
            boost::format("Value not found for Const node - %1%")
            % nodeDef.name()));
    }

    const tensorflow::TensorProto& tfTensor = nodeDef.attr().at("value").tensor();
    const tensorflow::TensorShapeProto& tfTensorShape = tfTensor.tensor_shape();
    const tensorflow::DataType tfDataType = ReadMandatoryNodeTypeAttribute(nodeDef, "dtype");

    const auto GetDimensionSize = [](auto& d) { return d.size(); };

    std::vector<unsigned int> dimensionSizes;
    std::transform(tfTensorShape.dim().begin(), tfTensorShape.dim().end(),
        std::back_inserter(dimensionSizes), GetDimensionSize);

    // Calculate number of elements
    const DataType dataType = ConvertTfTensorDataType(tfDataType);
    unsigned int numElements = 0U;

    if (!dimensionSizes.empty())
    {
        numElements = std::accumulate(dimensionSizes.begin(), dimensionSizes.end(),
                                      1U, std::multiplies<unsigned int>());
    }

    std::vector<int8_t> tensorData;

    // Get tensor data from the list of values attribute
    if (tfTensor.tensor_content().empty())
    {
        InvokeParseFunction<ParseTfTensorValueList>::Result<void>(dataType, tfTensor, numElements, tensorData);

        // If the tensor shape is not defined, but there is a value list, then interpret the data as a 1D
        // tensor of the provided number of elements
        if (numElements == 0)
        {
            const unsigned int tfNumElements = static_cast<unsigned int>(tensorData.size()) / GetDataTypeSize(dataType);
            dimensionSizes.push_back(tfNumElements);
        }
    }
    // Get tensor data from tensor content attribute
    else
    {
        tensorData.assign(tfTensor.tensor_content().begin(), tfTensor.tensor_content().end());

        // Check if a tensor shape is defined for the tensor content
        if (numElements == 0)
        {
            throw ParseException(boost::str(
                boost::format("No tensor shape found for Const node - %1%")
                % nodeDef.name()));
        }
    }

    // Const node requires at least a list of values or a content attribute
    if (tensorData.empty())
    {
        throw ParseException(boost::str(
            boost::format("No tensor data found for Const node - %1%")
            % nodeDef.name()));
    }

    const TensorInfo tensorInfo(static_cast<unsigned int>(dimensionSizes.size()), dimensionSizes.data(), dataType);

    // If we have a list of values, then the length of the list must be
    // less than or equal to the number of elements implied by the shape argument
    if (tensorData.size() > tensorInfo.GetNumBytes())
    {
        throw ParseException(boost::str(
            boost::format("Number of elements (%1%) should be less than or equal \
            to the number of elements implied by the shape argument (%2%) for Const node - %3%")
            % (tensorData.size() / GetDataTypeSize(dataType))
            % tensorInfo.GetNumElements()
            % nodeDef.name()));
    }

    return InvokeParseFunction<MakeTfOperation<ParsedConstTfOperation>>::Result<ParsedTfOperationPtr>(
        dataType, this, nodeDef, tensorData, tensorInfo);
}

template<typename Type>
bool TfParser::HasParsedConstTensor(const std::string & nodeName) const
{
    auto it = m_ParsedTfOperations.find(nodeName);
    if (it == m_ParsedTfOperations.end() ||
        dynamic_cast<ParsedConstTfOperation<Type>*>(it->second.get()) == nullptr)
    {
        return false;
    }
    else
    {
        return true;
    }
}

ParsedTfOperationPtr TfParser::ParseConv2D(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (!HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports Convolution layers with constant weights");
    }
    ParsedConstTfOperation<float>* weightNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<float> *>(inputs[1].m_IndexedValue);

    std::string paddingString = ReadMandatoryNodeStringAttribute(nodeDef, "padding");
    std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
    std::vector<uint32_t> strides = ReadMandatoryNodeUint32ListAttribute(nodeDef, "strides");

    // read the dilations, if present - only [1,1,1,1] (the default) is supported
    std::vector<uint32_t> dilations = ReadOptionalNodeUint32ListAttribute(nodeDef, "dilations");
    if (!dilations.empty())
    {
        for (auto dilation : dilations)
        {
            if (dilation != 1u)
            {
                throw ParseException("ArmNN only supports Convolution layers with dilations [1,1,1,1]");
            }
        }
    }

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = false;

    if (dataFormat == "NHWC")
    {
        desc.m_StrideX = strides[2];
        desc.m_StrideY = strides[1];
        // Swizzle input to supported memory layout
        inputTensorInfo = armnnUtils::Permuted(inputSlot.GetTensorInfo(), NHWCToArmNN);
    }
    else if (dataFormat == "NCHW")
    {
        desc.m_StrideX = strides[3];
        desc.m_StrideY = strides[2];
    }
    else
    {
        throw ParseException("Unsupported data format passed for Conv2D. Only NHWC and NCHW supported");
    }

    uint32_t inputHeight = inputTensorInfo.GetShape()[2];
    uint32_t inputWidth = inputTensorInfo.GetShape()[3];

    std::vector<float> outputTensorData;

    ConstTensor weightTensor = weightNode->GetConstTensor(true, outputTensorData);

    uint32_t weightHeight = weightTensor.GetShape()[2];
    uint32_t weightWidth = weightTensor.GetShape()[3];

    bool padding = false;
    TensorInfo outputInfo;
    if (paddingString == "SAME")
    {
        padding = true;
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                  weightTensor.GetShape()[0],
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputHeight) /
                                      static_cast<float>(desc.m_StrideY))),
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputWidth) /
                                      static_cast<float>(desc.m_StrideX)))
                                }, DataType::Float32);
    }
    else if (paddingString == "VALID")
    {
        padding = false;
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                  weightTensor.GetShape()[0],
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputHeight - weightHeight + 1) /
                                      static_cast<float>(desc.m_StrideY))),
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputWidth - weightWidth + 1) /
                                      static_cast<float>(desc.m_StrideX)))
                                }, DataType::Float32);
    }
    else
    {
        throw ParseException("Only 'SAME' and 'VALID' padding supported");
    }

    CalcPadding(inputHeight, weightHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, padding);
    CalcPadding(inputWidth, weightWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, padding);

    IConnectableLayer* layer = m_Network->AddConvolution2dLayer(desc, weightTensor, nodeDef.name().c_str());
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    if (dataFormat == "NHWC")
    {
        layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, nodeDef.name());
    }
    else
    {
        inputSlot.Connect(layer->GetInputSlot(0));
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseDepthwiseConv2D(const tensorflow::NodeDef& nodeDef,
                                                   const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (!HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports Depthwise Convolution layers with constant weights");
    }
    ParsedConstTfOperation<float>* weightNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<float> *>(inputs[1].m_IndexedValue);


    std::string paddingString = ReadMandatoryNodeStringAttribute(nodeDef, "padding");
    std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
    std::vector<uint32_t> strides = ReadMandatoryNodeUint32ListAttribute(nodeDef, "strides");

    DepthwiseConvolution2dDescriptor desc;
    desc.m_BiasEnabled = false;

    if (dataFormat == "NHWC")
    {
        desc.m_StrideX = strides[2];
        desc.m_StrideY = strides[1];
        // Swizzle input to supported memory layout
        inputTensorInfo = armnnUtils::Permuted(inputSlot.GetTensorInfo(), NHWCToArmNN);
    }
    else if (dataFormat == "NCHW")
    {
        desc.m_StrideX = strides[3];
        desc.m_StrideY = strides[2];
    }
    else
    {
        throw ParseException("Unsupported data format passed for DepthwiseConv2dNative. Only NHWC and NCHW supported");
    }

    uint32_t inputHeight = inputTensorInfo.GetShape()[2];
    uint32_t inputWidth = inputTensorInfo.GetShape()[3];

    std::vector<float> outputTensorData;

    ConstTensor weightTensor = weightNode->GetConstTensor(true, outputTensorData);

    uint32_t weightHeight = weightTensor.GetShape()[2];
    uint32_t weightWidth = weightTensor.GetShape()[3];

    bool padding = false;
    TensorInfo outputInfo;
    if (paddingString == "SAME")
    {
        padding = true;
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                weightTensor.GetShape()[0] * weightTensor.GetShape()[1],
                                static_cast<uint32_t>(ceil(
                                    static_cast<float>(inputHeight) /
                                    static_cast<float>(desc.m_StrideY))),
                                static_cast<uint32_t>(ceil(
                                    static_cast<float>(inputWidth) /
                                    static_cast<float>(desc.m_StrideX)))
                                }, DataType::Float32);
    }
    else if (paddingString == "VALID")
    {
        padding = false;
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                weightTensor.GetShape()[0] * weightTensor.GetShape()[1],
                                static_cast<uint32_t>(ceil(
                                    static_cast<float>(inputHeight - weightHeight + 1) /
                                    static_cast<float>(desc.m_StrideY))),
                                static_cast<uint32_t>(ceil(
                                    static_cast<float>(inputWidth - weightWidth + 1) /
                                    static_cast<float>(desc.m_StrideX)))
                                }, DataType::Float32);
    }
    else
    {
        throw ParseException("Only 'SAME' and 'VALID' padding supported");
    }

    CalcPadding(inputHeight, weightHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, padding);
    CalcPadding(inputWidth, weightWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, padding);

    IConnectableLayer* layer = m_Network->AddDepthwiseConvolution2dLayer(desc, weightTensor, nodeDef.name().c_str());
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    if (dataFormat == "NHWC")
    {
        layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, nodeDef.name());
    }
    else
    {
        inputSlot.Connect(layer->GetInputSlot(0));
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseFusedBatchNorm(const tensorflow::NodeDef& nodeDef,
                                                   const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 5);

    if (!HasParsedConstTensor<float>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant scale");
    }
    ParsedConstTfOperation<float>* scaleNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<float> *>(inputs[1].m_IndexedValue);

    if (!HasParsedConstTensor<float>(inputs[2].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant offset");
    }
    ParsedConstTfOperation<float>* offsetNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<float> *>(inputs[2].m_IndexedValue);

    if (!HasParsedConstTensor<float>(inputs[3].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant mean");
    }
    ParsedConstTfOperation<float>* meanNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<float> *>(inputs[3].m_IndexedValue);

    if (!HasParsedConstTensor<float>(inputs[4].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant variance");
    }
    ParsedConstTfOperation<float>* varianceNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<float> *>(inputs[4].m_IndexedValue);

    // The descriptor only has the epsilon attribute
    BatchNormalizationDescriptor desc;
    desc.m_Eps = ReadMandatoryNodeFloatAttribute(nodeDef, "epsilon");

    // data for the parsed tensor args (scale, offset, mean, variance) must be stored locally until the layer is added
    std::vector<float> scaleTensorData;
    ConstTensor scaleTensor = scaleNode->GetConstTensor(false, scaleTensorData);

    std::vector<float> offsetTensorData;
    ConstTensor offsetTensor = offsetNode->GetConstTensor(false, offsetTensorData);

    std::vector<float> meanTensorData;
    ConstTensor meanTensor = meanNode->GetConstTensor(false, meanTensorData);

    std::vector<float> varianceTensorData;
    ConstTensor varianceTensor = varianceNode->GetConstTensor(false, varianceTensorData);

    IConnectableLayer* layer = m_Network->AddBatchNormalizationLayer(desc,
                                                                     meanTensor,
                                                                     varianceTensor,
                                                                     offsetTensor,
                                                                     scaleTensor,
                                                                     nodeDef.name().c_str());

    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);

    const std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");

    if (dataFormat == "NHWC")
    {
        const TensorInfo outputTensorInfo = armnnUtils::Permuted(inputSlot.GetTensorInfo(), NHWCToArmNN);
        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
        layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, nodeDef.name());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(inputSlot.GetTensorInfo());
        inputSlot.Connect(layer->GetInputSlot(0));
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseConcat(const tensorflow::NodeDef& nodeDef,
                                           const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfConstNodeDef> nodes = GetTfInputNodes(nodeDef);
    // In tensorflow, we have the last input of the Concat layer as the axis for concatenation
    unsigned int numInputs = static_cast<unsigned int>(nodes.size());
    unsigned int numConcatView = numInputs - 1;

    OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), MaxNumOfTensorDimensions);
    std::vector<unsigned int>mergeDimSizes(MaxNumOfTensorDimensions, 0u);

    unsigned int mergeDim = 0;
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, numInputs);

    // The last input is the axis for concatenation
    if (!HasParsedConstTensor<int32_t>(inputs[numInputs - 1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports Concat with constant axis");
    }
    ParsedConstTfOperation<int32_t>* shapeNode =
            boost::polymorphic_downcast<ParsedConstTfOperation<int32_t>*>(inputs[numInputs - 1].m_IndexedValue);

    std::vector<int32_t> axisTensorData;
    ConstTensor axisTensor = shapeNode->GetConstTensor(false, axisTensorData);

    // This concatDim indicates the data format: 3 is the NHWC, 1 is the NCHW
    const unsigned int concatDimInput = static_cast<unsigned int>(axisTensorData[0]);

    // Armnn supports concatenation along the channel dimension for data format NHWC and NCHW
    if (concatDimInput == 0 || concatDimInput == 2)
    {
        throw ParseException("The dimension for concatenation is not supported by Armnn");
    }

    // This is the only concatDim we support in Armnn
    const unsigned int concatDim = 1;
    for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
    {
        // need to double check whether it should be
        IOutputSlot& inputSlot =
            inputs[viewIndex].m_IndexedValue->ResolveArmnnOutputSlot(inputs[viewIndex].m_Index);
        TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

        if (inputTensorInfo.GetNumDimensions() != MaxNumOfTensorDimensions)
        {
            throw ParseException("The number of dimensions for input tensors of the concatenation op should be 4");
        }

        if (concatDimInput == 3)
        {
            inputTensorInfo = armnnUtils::Permuted(inputTensorInfo, NHWCToArmNN);
        }

        for (unsigned int dim = 0; dim < MaxNumOfTensorDimensions; ++dim)
        {
            mergeDimSizes[dim] = inputTensorInfo.GetShape()[dim];
        }

        for (unsigned int j = 0; j < concatDim; ++j)
        {
            concatDescriptor.SetViewOriginCoord(viewIndex, j, 0);
        }

        concatDescriptor.SetViewOriginCoord(viewIndex, concatDim, mergeDim);
        mergeDim += mergeDimSizes[concatDim];

        for (unsigned int j = concatDim+1; j < MaxNumOfTensorDimensions; ++j)
        {
            concatDescriptor.SetViewOriginCoord(viewIndex, j, 0);
        }
    }

    mergeDimSizes[concatDim] = mergeDim;
    armnn::IConnectableLayer *layer = m_Network->AddMergerLayer(concatDescriptor, nodeDef.name().c_str());

    layer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(MaxNumOfTensorDimensions, mergeDimSizes.data(),
                                                            DataType::Float32));

    for (unsigned int v = 0; v < numConcatView; ++v)
    {
        IOutputSlot& inputSlot = inputs[v].m_IndexedValue->ResolveArmnnOutputSlot(inputs[v].m_Index);
        if (concatDimInput == 3)
        {
            IConnectableLayer* const swizzleLayer = AddSwizzleLayer(*m_Network, inputSlot, NHWCToArmNN,
                                                                    "swizzle_for-" + nodeDef.name());
            swizzleLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(v));
        }
        else
        {
            inputSlot.Connect(layer->GetInputSlot(v));
        }
    }

    if (concatDimInput == 3)
    {
        IConnectableLayer* const deswizzleLayer = AddSwizzleLayer(*m_Network, layer->GetOutputSlot(0), ArmNNToNHWC,
                                                                  "deswizzle_for-" + nodeDef.name());
        layer = deswizzleLayer;
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseShape(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    // Note: The Shape layer is handled in a special way, because:
    //        1. ARMNN doesn't support int32 tensors which it outputs
    //        2. ARMNN works with statically shaped tensors which are known at parse time
    //        3. because of 1. and 2. we treat the output of Shape as a temporary const int32
    //           tensor which may be used as an input to other ops, most likely a Reshape

    const tensorflow::DataType tfDataType = ReadMandatoryNodeTypeAttribute(nodeDef, "out_type");
    if (tfDataType != tensorflow::DT_INT32)
    {
        throw ParseException("Armnn only supports DT_INT32 as out_type");
    }

    const std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    const TensorInfo& prevLayerTensorInfo = prevLayerOutputSlot.GetTensorInfo();
    unsigned int prevLayerDimensions = prevLayerTensorInfo.GetNumDimensions();

    std::vector<int32_t> shapeTensorData;
    shapeTensorData.reserve(prevLayerDimensions);

    for (unsigned int i=0; i<prevLayerDimensions; ++i)
    {
        shapeTensorData.push_back(static_cast<int32_t>(prevLayerTensorInfo.GetShape()[i]));
    }

    TensorInfo shapeTensorInfo(1, &prevLayerDimensions, DataType::Signed32);

    return std::make_unique<ParsedConstTfOperation<int32_t>>(this,
                                                             nodeDef,
                                                             &shapeTensorData[0],
                                                             shapeTensorInfo);
}

ParsedTfOperationPtr TfParser::ParseReshape(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);
    ParsedTfOperation* inputNode = inputs[0].m_IndexedValue;

    if (!HasParsedConstTensor<int32_t>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports Reshape layers with constant shapes");
    }
    ParsedConstTfOperation<int32_t>* shapeNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

    armnn::IOutputSlot& prevLayerOutputSlot = inputNode->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

    std::vector<int32_t> shapeTensorData;
    ConstTensor shapeTensor = shapeNode->GetConstTensor(false, shapeTensorData);
    const TensorInfo outputTensorInfo = PrepareReshape(inputTensorInfo, shapeTensorData);

    TensorShape targetShape = outputTensorInfo.GetShape();
    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = targetShape;

    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, nodeDef.name().c_str());
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseResizeBilinear(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    if (!HasParsedConstTensor<int32_t>(inputs[1].m_IndexedValue->GetNode().name()))
    {
        throw ParseException("ArmNN only supports ResizeBilinear layers with constant sizes");
    }
    ParsedConstTfOperation<int32_t>* sizeNode =
        boost::polymorphic_downcast<ParsedConstTfOperation<int32_t>*>(inputs[1].m_IndexedValue);

    // Check the align_corners attribute is not set
    if (ReadOptionalNodeBoolAttribute(nodeDef, "align_corners", false))
    {
        throw ParseException("ArmNN only supports ResizeBilinear layers with align_corners set to false");
    }

    // data for the parsed tensor args (size) must be stored locally
    std::vector<int32_t> sizeTensorData;
    ConstTensor sizeTensor = sizeNode->GetConstTensor(false, sizeTensorData);

    // The descriptor only has target height and width attributes, which we get from the size tensor
    ResizeBilinearDescriptor desc;
    desc.m_TargetHeight = static_cast<uint32_t> (sizeTensorData[0]);
    desc.m_TargetWidth = static_cast<uint32_t> (sizeTensorData[1]);

    IConnectableLayer* layer = m_Network->AddResizeBilinearLayer(desc, nodeDef.name().c_str());

    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();
    // the input shape is always in BHWC format, this will be swizzled below; for now,
    // get the batch and channels to make up the ArmNN output shape with the target size
    unsigned int outBatch = inputTensorInfo.GetShape()[0];
    unsigned int outChannels = inputTensorInfo.GetShape()[3];
    unsigned int outHeight = desc.m_TargetHeight;
    unsigned int outWidth = desc.m_TargetWidth;
    TensorShape outShape({outBatch, outChannels, outHeight, outWidth});
    // The output DataType is always Float32, regardless of the input DataType
    const TensorInfo outputTensorInfo(outShape, armnn::DataType::Float32);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // TensorFlow ResizeBilinear input is always in BHWC format, so add swizzle and deswizzle layers
    layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, nodeDef.name());

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

TensorInfo OutputShapeOfSqueeze(const tensorflow::NodeDef& nodeDef, TensorInfo inputTensorInfo)
{
    BOOST_ASSERT(nodeDef.op() == "Squeeze");
    tensorflow::DataType tfDataType = ReadMandatoryNodeTypeAttribute(nodeDef, "T");

    DataType type;
    if (tfDataType == tensorflow::DT_FLOAT)
    {
        type = DataType::Float32;
    }
    else if (tfDataType == tensorflow::DT_INT32)
    {
        type = DataType::Signed32;
    }
    else
    {
        throw ParseException(boost::str(
                boost::format("Unsupported DataType %1% for Squeeze operation")
                % tensorflow::DataType_Name(tfDataType)));
    }

    std::vector<uint32_t> squeezeDims = ReadOptionalNodeUint32ListAttribute(nodeDef, "squeeze_dims");
    if (squeezeDims.empty())
    {
        for(unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); i++)
        {
            if (inputTensorInfo.GetShape()[i] == 1)
            {
                squeezeDims.push_back(i);
            }
        }
    }

    std::vector<uint32_t> outputDims;
    for(unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); i++)
    {
        bool includeDimension = (std::find(squeezeDims.begin(), squeezeDims.end(), i) == squeezeDims.end());
        if (includeDimension)
        {
            outputDims.push_back(inputTensorInfo.GetShape()[i]);
        }
    }

    if (outputDims.size() > 4)
    {
        throw ParseException("Unsupported shape for Squeeze");
    }

    TensorInfo outTensorInfo = TensorInfo(boost::numeric_cast<unsigned int>(outputDims.size()),
                                          outputDims.data(),
                                          type);

    return outTensorInfo;
}

ParsedTfOperationPtr TfParser::ParseSqueeze(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

    TensorInfo outputInfo;
    outputInfo = OutputShapeOfSqueeze(nodeDef, inputTensorInfo);

    ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputInfo.GetShape();
    IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, nodeDef.name().c_str());
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseLrn(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    NormalizationDescriptor normalizationDescriptor;
    normalizationDescriptor.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
    normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Across;
    normalizationDescriptor.m_Alpha = ReadMandatoryNodeFloatAttribute(nodeDef, "alpha");
    normalizationDescriptor.m_Beta = ReadMandatoryNodeFloatAttribute(nodeDef, "beta");
    normalizationDescriptor.m_K = ReadMandatoryNodeFloatAttribute(nodeDef, "bias");
    normalizationDescriptor.m_NormSize = ReadMandatoryNodeUint32Attribute(nodeDef, "depth_radius");

    // The window size must be an odd value. For a window size of (2 * n + 1), TensorFlow defines depth_radius = n.
    normalizationDescriptor.m_NormSize = normalizationDescriptor.m_NormSize * 2 + 1;

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);

    IConnectableLayer* layer = m_Network->AddNormalizationLayer(normalizationDescriptor,
        nodeDef.name().c_str());

    const TensorInfo permutedInfo = armnnUtils::Permuted(prevLayerOutputSlot.GetTensorInfo(), NHWCToArmNN);
    layer->GetOutputSlot(0).SetTensorInfo(permutedInfo);

    layer = SwizzleInDeswizzleOut(*m_Network, prevLayerOutputSlot, *layer, nodeDef.name());

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

/// An ParsedTfOperation for a MatMul node.
/// Creation of the armnn FullyConnected layer is deferred until it is actually needed, because MatMul nodes are
/// often used for the first part of a biased FullyConnected (MatMul followed by Add) and in these cases armnn doesn't
/// need a separate layer for the MatMul.
class ParsedMatMulTfOperation : public DeferredSingleLayerParsedTfOperation
{
public:
    ParsedMatMulTfOperation(TfParser* parser, const tensorflow::NodeDef& node)
        : DeferredSingleLayerParsedTfOperation(parser, node)
    {
    }

    void CreateLayerDeferred() override
    {
        BOOST_ASSERT(m_Layer == nullptr);
        m_Layer = m_Parser->AddFullyConnectedLayer(m_Node, nullptr, m_Node.name().c_str());
    }
};

ParsedTfOperationPtr TfParser::ParseMatMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    // Defer the creation of the layer (see ParsedMatMulTfOperation).
    return std::make_unique<ParsedMatMulTfOperation>(this, nodeDef);
}

ParsedTfOperationPtr TfParser::ParseMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IConnectableLayer* const layer = m_Network->AddMultiplicationLayer(nodeDef.name().c_str());
    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    auto const input0NumDims = input0Slot->GetTensorInfo().GetNumDimensions();
    auto const input1NumDims = input1Slot->GetTensorInfo().GetNumDimensions();

    if (input0NumDims < input1NumDims)
    {
        const bool isNHWC = true;
        input0Slot = BroadcastForAddandMul(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
    }
    if (input1NumDims < input0NumDims)
    {
        const bool isNHWC = true;
        input1Slot = BroadcastForAddandMul(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
    }

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    if (input0NumDims < input1NumDims)
    {
        layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
    }
    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParsePlaceholder(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 0);

    const LayerBindingId layerId = boost::numeric_cast<LayerBindingId>(m_NetworkInputsBindingInfo.size());

    auto it = m_InputShapes.find(nodeDef.name());
    if (it == m_InputShapes.end())
    {
        throw ParseException("Missing input shape for Placeholder '" + nodeDef.name() + "'");
    }
    TensorInfo tensorInfo(it->second, DataType::Float32);

    IConnectableLayer* const layer = m_Network->AddInputLayer(layerId, nodeDef.name().c_str());

    layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    TrackInputBinding(layer, layerId, tensorInfo);

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseRelu(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::ReLu;
    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseRelu6(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::BoundedReLu;
    activationDesc.m_A = 6.0f;
    activationDesc.m_B = 0.0f;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseSigmoid(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::Sigmoid;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseSoftmax(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    SoftmaxDescriptor softmaxDescriptor;
    IConnectableLayer* const layer = m_Network->AddSoftmaxLayer(softmaxDescriptor, nodeDef.name().c_str());

    IOutputSlot& prevLayerSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    prevLayerSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(prevLayerSlot.GetTensorInfo());

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseSoftplus(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::SoftReLu;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::ParseTanh(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    boost::ignore_unused(graphDef);

    ActivationDescriptor activationDesc;
    activationDesc.m_Function = ActivationFunction::TanH;
    activationDesc.m_A = 1.0f;
    activationDesc.m_B = 1.0f;

    return AddActivationLayer(nodeDef, activationDesc);
}

ParsedTfOperationPtr TfParser::AddActivationLayer(const tensorflow::NodeDef& nodeDef,
    ActivationDescriptor& activationDesc)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);

    IConnectableLayer* const layer = m_Network->AddActivationLayer(activationDesc, nodeDef.name().c_str());

    IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(prevLayerOutputSlot.GetTensorInfo());
    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::ParseMaxPool(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    return ParsePooling2d(nodeDef, graphDef, PoolingAlgorithm::Max);
}

ParsedTfOperationPtr TfParser::ParseAvgPool(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef)
{
    return ParsePooling2d(nodeDef, graphDef, PoolingAlgorithm::Average);
}

ParsedTfOperationPtr TfParser::ParsePooling2d(const tensorflow::NodeDef& nodeDef,
    const tensorflow::GraphDef& graphDef, PoolingAlgorithm pooltype)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 1);
    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

    if (inputs.size() != 1)
    {
        throw ParseException("2D Pooling expects one input!");
    }

    std::string paddingString = ReadMandatoryNodeStringAttribute(nodeDef, "padding");
    std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
    std::vector<uint32_t> strides = ReadMandatoryNodeUint32ListAttribute(nodeDef, "strides");
    std::vector<uint32_t> ksize = ReadMandatoryNodeUint32ListAttribute(nodeDef, "ksize"); // size of pool windows

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType = pooltype;
    pooling2dDescriptor.m_PaddingMethod = PaddingMethod::Exclude;
    pooling2dDescriptor.m_OutputShapeRounding = OutputShapeRounding::Floor;

    if (dataFormat == "NHWC")
    {
        pooling2dDescriptor.m_StrideX    = strides[2];
        pooling2dDescriptor.m_StrideY    = strides[1];
        pooling2dDescriptor.m_PoolWidth  = ksize[2];
        pooling2dDescriptor.m_PoolHeight = ksize[1];
        // Swizzle input to supported memory layout
        inputTensorInfo = armnnUtils::Permuted(inputSlot.GetTensorInfo(), NHWCToArmNN);
    }
    else if (dataFormat == "NCHW")
    {
        pooling2dDescriptor.m_StrideX    = strides[3];
        pooling2dDescriptor.m_StrideY    = strides[2];
        pooling2dDescriptor.m_PoolWidth  = ksize[3];
        pooling2dDescriptor.m_PoolHeight = ksize[2];
    }
    else
    {
        throw ParseException("Only NHWC or NCHW supported for Pooling2d");
    }

    uint32_t inputHeight = inputTensorInfo.GetShape()[2];
    uint32_t inputWidth = inputTensorInfo.GetShape()[3];

    bool padding = false;
    TensorInfo outputInfo;
    if (paddingString == "SAME")
    {
        padding = true;
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                  inputTensorInfo.GetShape()[1],
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputHeight) /
                                      static_cast<float>(pooling2dDescriptor.m_StrideY))),
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputWidth) /
                                      static_cast<float>(pooling2dDescriptor.m_StrideX)))
                                }, DataType::Float32);
    }
    else if (paddingString == "VALID")
    {
        padding = false;
        outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                  inputTensorInfo.GetShape()[1],
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputHeight - pooling2dDescriptor.m_PoolHeight + 1) /
                                      static_cast<float>(pooling2dDescriptor.m_StrideY))),
                                  static_cast<uint32_t>(ceil(
                                      static_cast<float>(inputWidth - pooling2dDescriptor.m_PoolWidth + 1) /
                                      static_cast<float>(pooling2dDescriptor.m_StrideX)))
                                }, DataType::Float32);
    }
    else
    {
        throw ParseException("Only 'SAME' and 'VALID' padding supported");
    }

    CalcPadding(inputWidth, pooling2dDescriptor.m_PoolWidth, pooling2dDescriptor.m_StrideX,
                    pooling2dDescriptor.m_PadLeft, pooling2dDescriptor.m_PadRight, padding);
    CalcPadding(inputHeight, pooling2dDescriptor.m_PoolHeight, pooling2dDescriptor.m_StrideY,
                    pooling2dDescriptor.m_PadTop, pooling2dDescriptor.m_PadBottom, padding);


    IConnectableLayer* layer = m_Network->AddPooling2dLayer(pooling2dDescriptor, nodeDef.name().c_str());
    if (layer == nullptr)
    {
        throw ParseException("Failed to add pooling2d layer");
    }

    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    if (dataFormat == "NHWC")
    {
        layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, nodeDef.name());
    }
    else
    {
        inputSlot.Connect(layer->GetInputSlot(0));
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

ParsedTfOperationPtr TfParser::AddAdditionLayer(const tensorflow::NodeDef& nodeDef, bool isBiasAdd)
{
    std::vector<OutputOfParsedTfOperation> inputs = GetInputParsedTfOperationsChecked(nodeDef, 2);

    IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
    IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

    const TensorInfo& input0Info = input0Slot->GetTensorInfo();
    const TensorInfo& input1Info = input1Slot->GetTensorInfo();

    if (isBiasAdd)
    {
        // BiasAdd takes bias as a 1D tensor. We need to add a reshape layer to create a 4D tensor
        // with the same data in the correct dimension for broadcast in addition.
        if(input1Info.GetNumDimensions() != 1)
        {
            throw ParseException("Unsupported bias for BiasAdd. It should be a 1D vector.");
        }

        const std::string dataFormat = ReadMandatoryNodeStringAttribute(nodeDef, "data_format");
        const bool isNHWC = (dataFormat == "NHWC");
        const bool isNCHW = (dataFormat == "NCHW");

        if (!isNHWC && ! isNCHW)
        {
            throw ParseException("Only NHWC or NCHW supported for BiasAdd");
        }

        input1Slot = BroadcastForAddandMul(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
    }
    else
    {
        if (input0Info.GetNumDimensions() == 1)
        {
            const bool isNHWC = true;
            input0Slot = BroadcastForAddandMul(input1Slot, input0Slot, isNHWC, *m_Network, nodeDef);
        }

        if (input1Info.GetNumDimensions() == 1)
        {
            const bool isNHWC = true;
            input1Slot = BroadcastForAddandMul(input0Slot, input1Slot, isNHWC, *m_Network, nodeDef);
        }
    }

    IConnectableLayer* const layer = m_Network->AddAdditionLayer(nodeDef.name().c_str());

    input0Slot->Connect(layer->GetInputSlot(0));
    input1Slot->Connect(layer->GetInputSlot(1));

    if (input0Info.GetNumDimensions() == 1 && isBiasAdd == false)
    {
        layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
    }
    else
    {
        layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
    }

    return std::make_unique<SingleLayerParsedTfOperation>(this, nodeDef, layer);
}

IConnectableLayer* TfParser::AddFullyConnectedLayer(const tensorflow::NodeDef& matMulNodeDef,
    const tensorflow::NodeDef* addNodeDef, const char* armnnLayerName)
{
    // find bias const (if applicable)
    ParsedConstTfOperation<float>* biasNode = nullptr;
    if (addNodeDef != nullptr)
    {
        std::vector<OutputOfParsedTfOperation> addInputs = GetInputParsedTfOperationsChecked(*addNodeDef, 2);
        // find our inputs
        if (HasParsedConstTensor<float>(addInputs[0].m_IndexedValue->GetNode().name()))
        {
            biasNode = boost::polymorphic_downcast<ParsedConstTfOperation<float>*>(addInputs[0].m_IndexedValue);
        }
        else if (HasParsedConstTensor<float>(addInputs[1].m_IndexedValue->GetNode().name()))
        {
            biasNode = boost::polymorphic_downcast<ParsedConstTfOperation<float>*>(addInputs[1].m_IndexedValue);
        }
        else
        {
            throw ParseException("ArmNN only supports fully connected layers with constant bias");
        }
    }

    // find matmul inputs
    ParsedConstTfOperation<float>* weightNode = nullptr;
    ParsedTfOperation* inputNode  = nullptr;
    unsigned int inputIdx = 0;
    std::vector<OutputOfParsedTfOperation> mulInputs = GetInputParsedTfOperationsChecked(matMulNodeDef, 2);
    if (HasParsedConstTensor<float>(mulInputs[0].m_IndexedValue->GetNode().name()))
    {
        weightNode = boost::polymorphic_downcast<ParsedConstTfOperation<float>*>(mulInputs[0].m_IndexedValue);
        inputNode = mulInputs[1].m_IndexedValue;
        inputIdx = mulInputs[1].m_Index;
    }
    else if (HasParsedConstTensor<float>(mulInputs[1].m_IndexedValue->GetNode().name()))
    {
        weightNode = boost::polymorphic_downcast<ParsedConstTfOperation<float>*>(mulInputs[1].m_IndexedValue);
        inputNode = mulInputs[0].m_IndexedValue;
        inputIdx = mulInputs[0].m_Index;
    }
    else
    {
        throw ParseException("ArmNN only supports fully connected layers with constant weights");
    }

    std::vector<float> weightTensorData;
    // handle weight
    ConstTensor weights = weightNode->GetConstTensor(false, weightTensorData);

    FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = addNodeDef != nullptr;

    IConnectableLayer* layer = nullptr;
    // make the layer
    if (addNodeDef != nullptr)
    {
        std::vector<float> biasTensorData;
        ConstTensor biases = biasNode->GetConstTensor(false, biasTensorData);

        if (weights.GetShape()[1] != biases.GetShape()[0])
        {
            throw ParseException("shape of matmul and bias do not match");
        }

        layer = m_Network->AddFullyConnectedLayer(desc, weights, biases, armnnLayerName);
    }
    else
    {
        layer = m_Network->AddFullyConnectedLayer(desc, weights, armnnLayerName);
    }

    BOOST_ASSERT(layer != nullptr);

    inputNode->ResolveArmnnOutputSlot(inputIdx).Connect(layer->GetInputSlot(0));
    unsigned int batches = inputNode->ResolveArmnnOutputSlot(inputIdx).GetTensorInfo().GetShape()[0];

    // handle output
    TensorInfo outputInfo({ batches, weights.GetShape()[1] }, DataType::Float32);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    return layer;
}

void TfParser::LoadNodeDef(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef)
{
    // get the type of the node (assume float)
    tensorflow::DataType type = tensorflow::DT_FLOAT;
    if (nodeDef.attr().count("T") != 0)
    {
        auto attr = nodeDef.attr().at("T");
        type      = attr.type();
    }
    else if (nodeDef.attr().count("dtype") != 0)
    {
        auto attr = nodeDef.attr().at("dtype");
        type      = attr.type();
    }

    if (type != tensorflow::DT_FLOAT && nodeDef.op() != "Const")
    {
        throw ParseException("Currently only FLOAT is supported for tensorflow nodes (apart from Const)");
    }

    const std::string& operation = nodeDef.op();
    auto it = ms_OperationNameToParsingFunctions.find(operation);
    if (it != ms_OperationNameToParsingFunctions.end())
    {
        auto func = it->second;
        ParsedTfOperationPtr parsedTfOperation = (this->*func)(nodeDef, graphDef);
        ParsedTfOperation* parsedTfOperationRaw = parsedTfOperation.get();

        // Store the parsed operation so that dependent layers can connect to it
        auto it = m_ParsedTfOperations.find(nodeDef.name());
        if (it != m_ParsedTfOperations.end())
        {
            throw ParseException(boost::str(boost::format("Name %1% used by more than one node") % nodeDef.name()));
        }
        m_ParsedTfOperations[nodeDef.name()] = std::move(parsedTfOperation);

        // If this node was requested as an output from the network then add an ArmNN output layer
        if (std::find(m_RequestedOutputs.begin(), m_RequestedOutputs.end(), nodeDef.name()) !=
            m_RequestedOutputs.end())
        {
            auto outId = ParseOutputId(nodeDef.name());
            const LayerBindingId layerId = boost::numeric_cast<LayerBindingId>(m_NetworkOutputsBindingInfo.size());
            IOutputSlot& prevSlot = parsedTfOperationRaw->ResolveArmnnOutputSlot(outId.m_Index);

            TensorInfo tensorInfo = prevSlot.GetTensorInfo();

            IConnectableLayer* outputLayer = m_Network->AddOutputLayer(layerId, nodeDef.name().c_str());

            prevSlot.Connect(outputLayer->GetInputSlot(0));

            TrackOutputBinding(outputLayer, layerId, tensorInfo);
        }
    }
    else
    {
        throw ParseException(boost::str(
            boost::format("Unsupported operation %1% in tensorflow::GraphDef") % operation));
    }
}

void TfParser::LoadGraphDef(const tensorflow::GraphDef& graphDef)
{
    // add all nodes to our map
    m_NodesByName.clear();
    m_NetworkInputsBindingInfo.clear();
    m_NetworkOutputsBindingInfo.clear();

    for (int i = 0; i < graphDef.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = graphDef.node(i);
        m_NodesByName[node.name()]      = &node;
    }

    // Find the output nodes the user requested
    std::vector<const tensorflow::NodeDef*> targetNodes;
    for (const std::string& requestedOutputName : m_RequestedOutputs)
    {
        auto nodeIt = m_NodesByName.find(requestedOutputName);
        if (nodeIt == m_NodesByName.end())
        {
            throw ParseException("Couldn't find requested output node '" + requestedOutputName + "' in graph");
        }
        targetNodes.push_back(nodeIt->second);
    }

    // Sort them into a linear ordering such that all inputs of a node are before the node itself
    std::vector<const tensorflow::NodeDef*> sortedNodes;
    if (!armnnUtils::GraphTopologicalSort<const tensorflow::NodeDef*>(
        targetNodes,
        [this](const tensorflow::NodeDef* node)
        {
            auto outputs = GetTfInputNodes(*node);
            std::vector<const tensorflow::NodeDef*> nodesOnly;
            for (const auto & o : outputs) {
                nodesOnly.push_back(o.m_IndexedValue);
            }
            return nodesOnly;
        },
        sortedNodes))
    {
        throw ParseException("Cycle detected in graph");
    }

    // Parse each node in order, knowing that all inputs of a node will be processed before the node itself
    for (const auto& it : sortedNodes)
    {
        const tensorflow::NodeDef& currentNode = *it;
        LoadNodeDef(currentNode, graphDef);
    }
}

INetworkPtr TfParser::CreateNetworkFromTextFile(const char* graphFile,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    FILE* fd = fopen(graphFile, "r");

    if (fd == nullptr)
    {
        std::stringstream error;
        error << "Graph file " << graphFile << " failed to open";
        throw FileNotFoundException(error.str());
    }

    // Parse the file into a message
    tensorflow::GraphDef graphDef;
    auto                 input   = new google::protobuf::io::FileInputStream(fileno(fd));
    bool                 success = google::protobuf::TextFormat::Parse(input, &graphDef);
    delete input;
    fclose(fd);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph file";
        throw ParseException(error.str());
    }

    return CreateNetworkFromGraphDef(graphDef, inputShapes, requestedOutputs);
}

INetworkPtr TfParser::CreateNetworkFromString(const char* protoText,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    // Parse the string into a message
    tensorflow::GraphDef graphDef;
    bool success = google::protobuf::TextFormat::ParseFromString(protoText, &graphDef);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse graph file";
        throw ParseException(error.str());
    }

    return CreateNetworkFromGraphDef(graphDef, inputShapes, requestedOutputs);
}

INetworkPtr TfParser::CreateNetworkFromBinaryFile(const char* graphFile,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    FILE* fd = fopen(graphFile, "rb");

    if (fd == nullptr)
    {
        std::stringstream error;
        error << "Graph file " << graphFile << " failed to open";
        throw FileNotFoundException(error.str());
    }

    // Parse the file into a message
    tensorflow::GraphDef graphDef;

    google::protobuf::io::FileInputStream  inStream(fileno(fd));
    google::protobuf::io::CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success = graphDef.ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        std::stringstream error;
        error << "Failed to parse protobuf file" << graphFile;
        throw ParseException(error.str());
    }

    return CreateNetworkFromGraphDef(graphDef, inputShapes, requestedOutputs);
}

INetworkPtr TfParser::CreateNetworkFromGraphDef(const tensorflow::GraphDef& graphDef,
    const std::map<std::string, TensorShape>& inputShapes,
    const std::vector<std::string>& requestedOutputs)
{
    m_Network = INetwork::Create();

    m_InputShapes = inputShapes;
    if (requestedOutputs.size() == 0)
    {
        throw ParseException("requestedOutputs must have at least one entry");
    }
    m_RequestedOutputs = requestedOutputs;

    try
    {
        LoadGraphDef(graphDef);
    }
    catch (const ParseException& e)
    {
        Cleanup();
        throw e;
    }

    Cleanup();

    return std::move(m_Network);
}

void TfParser::Cleanup()
{
    // cleanup, in case we reuse this parser
    m_InputShapes.clear();
    m_RequestedOutputs.clear();
    m_NodesByName.clear();
    m_ParsedTfOperations.clear();
}

BindingPointInfo TfParser::GetNetworkInputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "input", m_NetworkInputsBindingInfo);
}

BindingPointInfo TfParser::GetNetworkOutputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
}

std::pair<LayerBindingId, TensorInfo> TfParser::GetBindingInfo(const std::string& layerName,
    const char* bindingPointDesc,
    const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        throw InvalidArgumentException(boost::str(boost::format("Unknown %1% '%2%'") % bindingPointDesc % layerName));
    }
    return it->second;
}

void TfParser::TrackInputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, "input", m_NetworkInputsBindingInfo);
}

void TfParser::TrackOutputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, "output", m_NetworkOutputsBindingInfo);
}

void TfParser::TrackBindingPoint(IConnectableLayer* layer,
    LayerBindingId id,
    const TensorInfo& tensorInfo,
    const char* bindingPointDesc,
    std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    const std::string layerName = layer->GetName();
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        nameToBindingInfo[layerName] = std::make_pair(id, tensorInfo);
    }
    else
    {
        throw ParseException(boost::str(
            boost::format("Id %1% used by more than one %2% layer") % id % bindingPointDesc));
    }
}

} // namespace armnnTfParser
