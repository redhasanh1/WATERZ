/*
 * DCNv4 TensorRT Plugin Implementation
 *
 * Implements TensorRT IPluginV2DynamicExt interface for DCNv4 deformable convolution.
 *
 * Author: Claude Code
 * Target: NVIDIA RTX 4090 (Ada Lovelace, Compute Capability 8.9)
 */

#include "DCNv4Plugin.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

// Forward declaration of CUDA kernel (implemented in Phase 2G-2/2G-3)
extern "C" void dcnv4_forward_cuda(
    const void* input,        // [N, C, H, W] in FP16
    void* output,             // [N, C, H, W] in FP16
    void* workspace,          // Temp workspace
    int batch,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int pad,
    int group,
    float offset_scale,
    bool remove_center,
    cudaStream_t stream
);

// Forward declaration of CUDA kernel with offset input (Phase 2G-3)
extern "C" void dcnv4_forward_cuda_with_offsets(
    const void* input,        // [N, C, H, W] in FP16
    const void* offsets,      // [N, G*K*3, H, W] in FP16
    void* output,             // [N, C, H, W] in FP16
    void* workspace,          // Temp workspace
    int batch,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int pad,
    int group,
    float offset_scale,
    bool remove_center,
    cudaStream_t stream
);

namespace nvinfer1 {
namespace plugin {

// Helper functions for serialization
template<typename T>
void write(char*& buffer, const T& val) {
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

template<typename T>
T read(const char*& buffer) {
    T val;
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

//=============================================================================
// DCNv4Plugin Implementation
//=============================================================================

DCNv4Plugin::DCNv4Plugin(const std::string& name,
                         int channels,
                         int kernelSize,
                         int stride,
                         int pad,
                         int group,
                         float offsetScale,
                         bool removeCenter)
    : mLayerName(name)
    , mNamespace("")
    , mChannels(channels)
    , mKernelSize(kernelSize)
    , mStride(stride)
    , mPad(pad)
    , mGroup(group)
    , mOffsetScale(offsetScale)
    , mRemoveCenter(removeCenter)
    , mBatchSize(0)
    , mHeight(0)
    , mWidth(0)
    , mNumFrames(0)
    , mWorkspaceSize(0)
{
    // Validate parameters
    assert(channels > 0 && "channels must be positive");
    assert(kernelSize == 3 && "Only kernel_size=3 supported for now");
    assert(stride == 1 && "Only stride=1 supported for RFC Net");
    assert(pad == 1 && "Only pad=1 supported for RFC Net");
    assert(group >= 1 && group <= 8 && "group must be in [1, 8]");
}

DCNv4Plugin::DCNv4Plugin(const std::string& name, const void* data, size_t length)
    : mLayerName(name)
    , mNamespace("")
{
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    // Deserialize parameters
    mChannels = read<int>(d);
    mKernelSize = read<int>(d);
    mStride = read<int>(d);
    mPad = read<int>(d);
    mGroup = read<int>(d);
    mOffsetScale = read<float>(d);
    mRemoveCenter = read<bool>(d);

    // Initialize runtime parameters
    mBatchSize = 0;
    mHeight = 0;
    mWidth = 0;
    mNumFrames = 0;
    mWorkspaceSize = 0;

    assert(d == a + length && "Deserialization size mismatch");
}

//-----------------------------------------------------------------------------
// IPluginV2DynamicExt Methods
//-----------------------------------------------------------------------------

int DCNv4Plugin::getNbOutputs() const noexcept {
    // DCNv4 produces single output (aligned features)
    return 1;
}

DimsExprs DCNv4Plugin::getOutputDimensions(
    int outputIndex,
    const DimsExprs* inputs,
    int nbInputs,
    IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex == 0 && "DCNv4Plugin only has 1 output");
    assert(nbInputs == 2 && "DCNv4Plugin expects 2 inputs (features + offsets)");

    // Output shape = Input[0] shape (deformable conv preserves feature dimensions)
    // Input[0]: [N, C, H, W] or [N, C, T, H, W] (features)
    // Input[1]: [N, G*K*3, H, W] or [N, G*K*3, T, H, W] (offsets)
    DimsExprs output;
    output.nbDims = inputs[0].nbDims;
    for (int i = 0; i < output.nbDims; i++) {
        output.d[i] = inputs[0].d[i];
    }

    return output;
}

int DCNv4Plugin::initialize() noexcept {
    // No initialization needed (CUDA kernels are stateless)
    return 0;
}

void DCNv4Plugin::terminate() noexcept {
    // No cleanup needed
}

size_t DCNv4Plugin::getWorkspaceSize(
    const PluginTensorDesc* inputs,
    int nbInputs,
    const PluginTensorDesc* outputs,
    int nbOutputs) const noexcept
{
    assert(nbInputs == 2);  // features + offsets
    assert(nbOutputs == 1);

    // Get input dimensions
    const auto& inputDims = inputs[0].dims;

    // Calculate workspace size for format conversion and intermediate buffers
    // Input format: [N, C, H, W] or [N, C, T, H, W]
    int batch = inputDims.d[0];
    int channels = inputDims.d[1];
    int height, width, temporal;

    if (inputDims.nbDims == 4) {
        // [N, C, H, W]
        height = inputDims.d[2];
        width = inputDims.d[3];
        temporal = 1;
    } else if (inputDims.nbDims == 5) {
        // [N, C, T, H, W]
        temporal = inputDims.d[2];
        height = inputDims.d[3];
        width = inputDims.d[4];
    } else {
        return 0;  // Invalid dimensions
    }

    int spatial_size = height * width;

    // Workspace needed for DCNv4 forward pass (Phase 2G-3):
    // 1. Input buffer in NHWC format: [B, H, W, C]
    // 2. Offset tensor in reshaped format: [B, H*W, padded_offset_dim]
    // 3. Output buffer in NHWC format: [B, H_out, W_out, C]
    //
    // Note: Offsets are provided as input[1] in NCHW format [N, G*K*3, H, W]
    //       but need to be reshaped to [N, H*W, G*K*3] for CUDA kernel

    size_t elementSize = sizeof(float);  // Default FP32
    if (inputs[0].type == DataType::kHALF) {
        elementSize = sizeof(uint16_t);  // 2 bytes for FP16
    }

    // Calculate padded offset dimension
    int kernel_points = mKernelSize * mKernelSize;
    if (mRemoveCenter) {
        kernel_points -= 1;
    }
    int base_offset_dim = mGroup * kernel_points * 3;  // 3 = (offset_x, offset_y, attention_weight)
    int padded_offset_dim = ((base_offset_dim + 7) / 8) * 8;  // Round up to multiple of 8

    // Calculate output spatial dimensions (for stride=1, pad=1, kernel=3, output size equals input size)
    int height_out = (height + 2 * mPad - mKernelSize) / mStride + 1;
    int width_out = (width + 2 * mPad - mKernelSize) / mStride + 1;

    // Buffer sizes
    size_t input_nhwc_size = batch * temporal * spatial_size * channels * elementSize;
    size_t offset_reshaped_size = batch * temporal * spatial_size * padded_offset_dim * elementSize;
    size_t output_nhwc_size = batch * temporal * height_out * width_out * channels * elementSize;

    // Total workspace with 1.2x safety margin
    size_t total_workspace = input_nhwc_size + offset_reshaped_size + output_nhwc_size;
    total_workspace = static_cast<size_t>(total_workspace * 1.2);  // 20% safety margin

    return total_workspace;
}

int DCNv4Plugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    // Get input/output dimensions
    const auto& inputDims = inputDesc[0].dims;

    int batch = inputDims.d[0];
    int channels = inputDims.d[1];
    int height, width, temporal;

    if (inputDims.nbDims == 4) {
        // [N, C, H, W]
        height = inputDims.d[2];
        width = inputDims.d[3];
        temporal = 1;
    } else if (inputDims.nbDims == 5) {
        // [N, C, T, H, W]
        temporal = inputDims.d[2];
        height = inputDims.d[3];
        width = inputDims.d[4];
    } else {
        std::cerr << "[DCNv4Plugin] ERROR: Invalid input dimensions" << std::endl;
        return -1;
    }

    // Validate parameters
    if (channels != mChannels) {
        std::cerr << "[DCNv4Plugin] ERROR: Channel mismatch. Expected "
                  << mChannels << ", got " << channels << std::endl;
        return -1;
    }

    // Get input/output pointers
    // inputs[0]: Feature map [N, C, H, W]
    // inputs[1]: Offsets [N, G*K*3, H, W]
    const void* input_features = inputs[0];
    const void* input_offsets = inputs[1];
    void* output = outputs[0];

    // Call CUDA kernel with offset input (Phase 2G-3)
    try {
        dcnv4_forward_cuda_with_offsets(
            input_features,
            input_offsets,
            output,
            workspace,
            batch * temporal,  // Flatten batch and temporal dimensions
            channels,
            height,
            width,
            mKernelSize,
            mStride,
            mPad,
            mGroup,
            mOffsetScale,
            mRemoveCenter,
            stream
        );
    } catch (const std::exception& e) {
        std::cerr << "[DCNv4Plugin] CUDA kernel failed: " << e.what() << std::endl;
        return -1;
    }

    return 0;  // Success
}

size_t DCNv4Plugin::getSerializationSize() const noexcept {
    return sizeof(int) * 5 +      // mChannels, mKernelSize, mStride, mPad, mGroup
           sizeof(float) +         // mOffsetScale
           sizeof(bool);           // mRemoveCenter
}

void DCNv4Plugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);

    write(d, mChannels);
    write(d, mKernelSize);
    write(d, mStride);
    write(d, mPad);
    write(d, mGroup);
    write(d, mOffsetScale);
    write(d, mRemoveCenter);
}

bool DCNv4Plugin::supportsFormatCombination(
    int pos,
    const PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) noexcept
{
    assert(nbInputs == 2 && nbOutputs == 1);  // 2 inputs (features + offsets), 1 output
    assert(pos < 3);  // 2 inputs + 1 output

    const PluginTensorDesc& desc = inOut[pos];

    // Support FP16 (half precision) for RTX 4090 Ada Lovelace
    if (desc.type != DataType::kHALF) {
        return false;
    }

    // Must use linear (NCHW) format
    if (desc.format != TensorFormat::kLINEAR) {
        return false;
    }

    // All tensors (input features, input offsets, output) must have same type and format
    if (pos == 1 || pos == 2) {
        return desc.type == inOut[0].type && desc.format == inOut[0].format;
    }

    return true;
}

const char* DCNv4Plugin::getPluginType() const noexcept {
    return DCNV4_PLUGIN_NAME;
}

const char* DCNv4Plugin::getPluginVersion() const noexcept {
    return DCNV4_PLUGIN_VERSION;
}

void DCNv4Plugin::destroy() noexcept {
    delete this;
}

IPluginV2DynamicExt* DCNv4Plugin::clone() const noexcept {
    auto* plugin = new DCNv4Plugin(
        mLayerName,
        mChannels,
        mKernelSize,
        mStride,
        mPad,
        mGroup,
        mOffsetScale,
        mRemoveCenter
    );
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void DCNv4Plugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* DCNv4Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

DataType DCNv4Plugin::getOutputDataType(
    int index,
    const DataType* inputTypes,
    int nbInputs) const noexcept
{
    assert(index == 0);
    assert(nbInputs == 1);

    // Output type matches input type (FP16)
    return inputTypes[0];
}

void DCNv4Plugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    IGpuAllocator* gpuAllocator) noexcept
{
    // No context attachment needed for DCNv4
}

void DCNv4Plugin::detachFromContext() noexcept {
    // No context detachment needed
}

void DCNv4Plugin::configurePlugin(
    const DynamicPluginTensorDesc* in,
    int nbInputs,
    const DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept
{
    assert(nbInputs == 1 && nbOutputs == 1);

    // Store runtime parameters
    const auto& inputDims = in[0].desc.dims;

    mBatchSize = inputDims.d[0];

    if (inputDims.nbDims == 4) {
        // [N, C, H, W]
        mHeight = inputDims.d[2];
        mWidth = inputDims.d[3];
        mNumFrames = 1;
    } else if (inputDims.nbDims == 5) {
        // [N, C, T, H, W]
        mNumFrames = inputDims.d[2];
        mHeight = inputDims.d[3];
        mWidth = inputDims.d[4];
    }

    // Calculate workspace size
    size_t elementSize = (in[0].desc.type == DataType::kHALF) ? sizeof(uint16_t) : sizeof(float);
    int spatial_size = mHeight * mWidth;
    size_t conversionSize = mBatchSize * mNumFrames * spatial_size * mChannels * elementSize;
    mWorkspaceSize = conversionSize * 3;  // 3x for conversion + DCNv4 internals
}

//=============================================================================
// DCNv4PluginCreator Implementation
//=============================================================================

PluginFieldCollection DCNv4PluginCreator::mFC{};
std::vector<PluginField> DCNv4PluginCreator::mPluginAttributes;

DCNv4PluginCreator::DCNv4PluginCreator() {
    // Define plugin fields for ONNX parsing
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("pad", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("offset_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("remove_center", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DCNv4PluginCreator::getPluginName() const noexcept {
    return DCNV4_PLUGIN_NAME;
}

const char* DCNv4PluginCreator::getPluginVersion() const noexcept {
    return DCNV4_PLUGIN_VERSION;
}

const PluginFieldCollection* DCNv4PluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2DynamicExt* DCNv4PluginCreator::createPlugin(
    const char* name,
    const PluginFieldCollection* fc) noexcept
{
    // Parse plugin attributes from ONNX
    int channels = 128;        // Default for RFC Net
    int kernelSize = 3;
    int stride = 1;
    int pad = 1;
    int group = 1;
    float offsetScale = 1.0f;
    bool removeCenter = false;

    for (int i = 0; i < fc->nbFields; i++) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "channels")) {
            channels = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "kernel_size")) {
            kernelSize = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "stride")) {
            stride = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "pad")) {
            pad = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "group")) {
            group = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "offset_scale")) {
            offsetScale = *static_cast<const float*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "remove_center")) {
            removeCenter = *static_cast<const int*>(fc->fields[i].data) != 0;
        }
    }

    auto* plugin = new DCNv4Plugin(name, channels, kernelSize, stride, pad, group, offsetScale, removeCenter);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* DCNv4PluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept
{
    auto* plugin = new DCNv4Plugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void DCNv4PluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* DCNv4PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

//=============================================================================
// Plugin Registration
//=============================================================================

REGISTER_TENSORRT_PLUGIN(DCNv4PluginCreator);

} // namespace plugin
} // namespace nvinfer1
