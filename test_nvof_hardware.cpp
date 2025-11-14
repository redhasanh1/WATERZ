/**
 * NVOF Hardware Detection Test for RTX 5070 Blackwell
 *
 * Purpose: Verify if NVIDIA Hardware Optical Flow is available on RTX 5070
 *          before investing time in full NVOF integration for RAFT optimization.
 *
 * What this tests:
 * 1. Can we load the NVOF library (nvofapi64.dll)?
 * 2. Can we create an NVOF handle (NvCreateOpticalFlowCuda)?
 * 3. What error code do we get? (NV_OF_SUCCESS vs NV_OF_ERR_OF_NOT_AVAILABLE)
 * 4. If successful, what capabilities does the hardware support?
 *
 * Build:
 *   cl /EHsc test_nvof_hardware.cpp /I"Optical_Flow_SDK_5.0.7\NvOFInterface" ^
 *      /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
 *      /link "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64\cuda.lib"
 *
 * Run:
 *   test_nvof_hardware.exe
 */

#include <iostream>
#include <iomanip>
#include <windows.h>
#include <cuda.h>

// NVOF SDK headers
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

// Helper function to get error string
const char* getNvOFStatusString(NV_OF_STATUS status) {
    switch (status) {
        case NV_OF_SUCCESS: return "NV_OF_SUCCESS";
        case NV_OF_ERR_OF_NOT_AVAILABLE: return "NV_OF_ERR_OF_NOT_AVAILABLE (Hardware not supported)";
        case NV_OF_ERR_UNSUPPORTED_DEVICE: return "NV_OF_ERR_UNSUPPORTED_DEVICE";
        case NV_OF_ERR_DEVICE_DOES_NOT_EXIST: return "NV_OF_ERR_DEVICE_DOES_NOT_EXIST";
        case NV_OF_ERR_INVALID_PTR: return "NV_OF_ERR_INVALID_PTR";
        case NV_OF_ERR_OUT_OF_MEMORY: return "NV_OF_ERR_OUT_OF_MEMORY";
        case NV_OF_ERR_INVALID_PARAM: return "NV_OF_ERR_INVALID_PARAM";
        case NV_OF_ERR_INVALID_VERSION: return "NV_OF_ERR_INVALID_VERSION";
        case NV_OF_ERR_GENERIC: return "NV_OF_ERR_GENERIC";
        default: return "UNKNOWN_ERROR";
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " NVOF Hardware Detection Test" << std::endl;
    std::cout << " RTX 5070 Blackwell GPU" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Step 1: Initialize CUDA
    std::cout << "[1/5] Initializing CUDA..." << std::endl;
    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(cuResult, &errStr);
        std::cerr << "ERROR: Failed to initialize CUDA: " << (errStr ? errStr : "Unknown") << std::endl;
        return 1;
    }
    std::cout << "      OK: CUDA initialized" << std::endl;

    // Step 2: Get device info
    std::cout << std::endl << "[2/5] Querying GPU information..." << std::endl;
    int deviceCount = 0;
    cuResult = cuDeviceGetCount(&deviceCount);
    if (cuResult != CUDA_SUCCESS || deviceCount == 0) {
        std::cerr << "ERROR: No CUDA devices found" << std::endl;
        return 1;
    }

    CUdevice cuDevice;
    cuResult = cuDeviceGet(&cuDevice, 0);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "ERROR: Failed to get CUDA device" << std::endl;
        return 1;
    }

    char deviceName[256];
    cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice);

    int computeCapabilityMajor, computeCapabilityMinor;
    cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

    std::cout << "      GPU: " << deviceName << std::endl;
    std::cout << "      Compute Capability: SM " << computeCapabilityMajor << "." << computeCapabilityMinor << std::endl;

    // Step 3: Create CUDA context
    std::cout << std::endl << "[3/5] Creating CUDA context..." << std::endl;
    CUcontext cuContext;
    cuResult = cuCtxCreate(&cuContext, 0, cuDevice);
    if (cuResult != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(cuResult, &errStr);
        std::cerr << "ERROR: Failed to create CUDA context: " << (errStr ? errStr : "Unknown") << std::endl;
        return 1;
    }
    std::cout << "      OK: CUDA context created" << std::endl;

    // Step 4: Load NVOF library
    std::cout << std::endl << "[4/5] Loading NVOF library (nvofapi64.dll)..." << std::endl;

    // Try loading from SDK directory first
    HMODULE nvofLib = LoadLibraryA("Optical_Flow_SDK_5.0.7\\lib\\x64\\nvofapi64.dll");
    if (!nvofLib) {
        // Try system PATH
        nvofLib = LoadLibraryA("nvofapi64.dll");
    }

    if (!nvofLib) {
        DWORD error = GetLastError();
        std::cerr << "ERROR: Failed to load nvofapi64.dll (error code: " << error << ")" << std::endl;
        std::cerr << "       Make sure nvofapi64.dll is in the SDK lib/x64/ folder or system PATH" << std::endl;
        cuCtxDestroy(cuContext);
        return 1;
    }
    std::cout << "      OK: NVOF library loaded" << std::endl;

    // Get API function
    typedef NV_OF_STATUS (NVOFAPI* PFNNvOFAPICreateInstanceCuda)(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST* functionList);
    PFNNvOFAPICreateInstanceCuda NvOFAPICreateInstanceCuda =
        (PFNNvOFAPICreateInstanceCuda)GetProcAddress(nvofLib, "NvOFAPICreateInstanceCuda");

    if (!NvOFAPICreateInstanceCuda) {
        std::cerr << "ERROR: Failed to get NvOFAPICreateInstanceCuda function pointer" << std::endl;
        FreeLibrary(nvofLib);
        cuCtxDestroy(cuContext);
        return 1;
    }

    // Create API instance
    NV_OF_CUDA_API_FUNCTION_LIST functionList = {};
    NV_OF_STATUS status = NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &functionList);
    if (status != NV_OF_SUCCESS) {
        std::cerr << "ERROR: NvOFAPICreateInstanceCuda failed: " << getNvOFStatusString(status) << std::endl;
        FreeLibrary(nvofLib);
        cuCtxDestroy(cuContext);
        return 1;
    }
    std::cout << "      OK: NVOF API instance created" << std::endl;

    // Step 5: Create NVOF handle (THIS IS THE CRITICAL TEST)
    std::cout << std::endl << "[5/5] Creating NVOF handle (testing hardware availability)..." << std::endl;
    NvOFHandle nvofHandle = nullptr;
    status = functionList.nvCreateOpticalFlowCuda(cuContext, &nvofHandle);

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " RESULT" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    if (status == NV_OF_SUCCESS && nvofHandle != nullptr) {
        std::cout << "SUCCESS: NVOF hardware IS AVAILABLE on RTX 5070!" << std::endl;
        std::cout << std::endl;
        std::cout << "Status: " << getNvOFStatusString(status) << std::endl;
        std::cout << "Handle: " << nvofHandle << std::endl;
        std::cout << std::endl;
        std::cout << "Next steps:" << std::endl;
        std::cout << "  1. We can proceed with full NVOF Python wrapper integration" << std::endl;
        std::cout << "  2. This should provide faster optical flow than RAFT TensorRT" << std::endl;
        std::cout << "  3. Hardware-accelerated optical flow confirmed available" << std::endl;

        // Query capabilities
        std::cout << std::endl << "Querying NVOF capabilities..." << std::endl;

        NV_OF_CAPS capsParams[] = {
            NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES,
            NV_OF_CAPS_WIDTH_MIN,
            NV_OF_CAPS_WIDTH_MAX,
            NV_OF_CAPS_HEIGHT_MIN,
            NV_OF_CAPS_HEIGHT_MAX
        };

        const char* capsNames[] = {
            "Supported grid sizes",
            "Min width",
            "Max width",
            "Min height",
            "Max height"
        };

        for (int i = 0; i < sizeof(capsParams) / sizeof(capsParams[0]); i++) {
            uint32_t capsValue = 0;
            uint32_t capsSize = 1;
            NV_OF_STATUS capsStatus = functionList.nvOFGetCaps(nvofHandle, capsParams[i], &capsValue, &capsSize);
            if (capsStatus == NV_OF_SUCCESS) {
                std::cout << "  " << capsNames[i] << ": " << capsValue;
                if (i == 0) {
                    std::cout << " (bitfield: 1=1x1, 2=2x2, 4=4x4)";
                }
                std::cout << std::endl;
            }
        }

        // Cleanup
        functionList.nvOFDestroy(nvofHandle);

    } else {
        std::cout << "FAILURE: NVOF hardware NOT AVAILABLE on RTX 5070" << std::endl;
        std::cout << std::endl;
        std::cout << "Status: " << getNvOFStatusString(status) << std::endl;
        std::cout << std::endl;
        std::cout << "Analysis:" << std::endl;
        std::cout << "  - This confirms the DLSS 4 article claim" << std::endl;
        std::cout << "  - Blackwell removed hardware optical flow accelerator" << std::endl;
        std::cout << "  - NVIDIA replaced it with AI models for DLSS frame generation" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommendation:" << std::endl;
        std::cout << "  - Stick with current RAFT TensorRT optimization (3.64ms, 8.45x speedup)" << std::endl;
        std::cout << "  - Do NOT invest time in NVOF Python wrapper (won't work on Blackwell)" << std::endl;
        std::cout << "  - Focus on wiring RFCNet TensorRT into watermark.py instead" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    // Cleanup
    FreeLibrary(nvofLib);
    cuCtxDestroy(cuContext);

    return (status == NV_OF_SUCCESS) ? 0 : 1;
}
