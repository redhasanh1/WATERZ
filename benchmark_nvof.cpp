/**
 * NVOF Standalone Benchmark
 *
 * Purpose: Measure NVIDIA Hardware Optical Flow performance on RTX 5070
 *          Compare against RAFT TensorRT (currently 3.64ms mean latency)
 *
 * Test configuration:
 *   - Input: 480x480 grayscale frames (typical ProPainter size)
 *   - Grid size: 1x1 (per-pixel flow, highest quality)
 *   - Performance preset: SLOW (best quality to match RAFT)
 *   - Iterations: 100 warmup + 1000 timed
 *
 * Build:
 *   BUILD_NVOF_BENCHMARK.bat
 *
 * Expected result:
 *   If mean latency < 3.64ms: NVOF is faster → proceed with Python integration
 *   If mean latency >= 3.64ms: RAFT TensorRT is faster → skip NVOF integration
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <cmath>
#include <algorithm>
#include <windows.h>
#include <cuda.h>

// NVOF SDK low-level C API
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

// Error handling macros
#define CHECK_NVOF(call) \
    do { \
        NV_OF_STATUS status = (call); \
        if (status != NV_OF_SUCCESS) { \
            std::cerr << "NVOF error at " << __FILE__ << ":" << __LINE__ << " - status " << status << std::endl; \
            throw std::runtime_error("NVOF error"); \
        } \
    } while(0)

#define CHECK_CUDA(call) \
    do { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(result, &errStr); \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << (errStr ? errStr : "Unknown") << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Helper to compute statistics
struct BenchmarkStats {
    double mean;
    double std;
    double min;
    double max;
    double p50;
    double p95;
    double p99;
};

BenchmarkStats computeStats(std::vector<double>& latencies) {
    BenchmarkStats stats;

    // Sort for percentiles
    std::sort(latencies.begin(), latencies.end());

    // Mean
    double sum = 0.0;
    for (double lat : latencies) {
        sum += lat;
    }
    stats.mean = sum / latencies.size();

    // Std dev
    double sq_sum = 0.0;
    for (double lat : latencies) {
        sq_sum += (lat - stats.mean) * (lat - stats.mean);
    }
    stats.std = std::sqrt(sq_sum / latencies.size());

    // Min/max
    stats.min = latencies.front();
    stats.max = latencies.back();

    // Percentiles
    auto percentile = [&](double p) -> double {
        size_t idx = static_cast<size_t>(latencies.size() * p / 100.0);
        return latencies[idx];
    };

    stats.p50 = percentile(50.0);
    stats.p95 = percentile(95.0);
    stats.p99 = percentile(99.0);

    return stats;
}

void printStats(const char* name, const BenchmarkStats& stats) {
    std::cout << "\n" << name << " Performance:" << std::endl;
    std::cout << "  Mean Latency:   " << stats.mean << " ms" << std::endl;
    std::cout << "  Std Dev:        " << stats.std << " ms" << std::endl;
    std::cout << "  Min Latency:    " << stats.min << " ms" << std::endl;
    std::cout << "  Max Latency:    " << stats.max << " ms" << std::endl;
    std::cout << "  P50 (Median):   " << stats.p50 << " ms" << std::endl;
    std::cout << "  P95:            " << stats.p95 << " ms" << std::endl;
    std::cout << "  P99:            " << stats.p99 << " ms" << std::endl;
    std::cout << "  Throughput:     " << (1000.0 / stats.mean) << " fps" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " NVOF Hardware Optical Flow Benchmark" << std::endl;
    std::cout << " RTX 5070 Blackwell GPU" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Configuration
    const uint32_t WIDTH = 480;
    const uint32_t HEIGHT = 480;
    const uint32_t GRID_SIZE = 1;  // Per-pixel flow
    const int WARMUP_ITERS = 100;
    const int TIMED_ITERS = 1000;

    try {
        // Initialize CUDA
        std::cout << "[1/7] Initializing CUDA..." << std::endl;
        CHECK_CUDA(cuInit(0));

        CUdevice cuDevice;
        CHECK_CUDA(cuDeviceGet(&cuDevice, 0));

        char deviceName[256];
        cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice);
        std::cout << "      GPU: " << deviceName << std::endl;

        CUcontext cuContext;
        CHECK_CUDA(cuCtxCreate(&cuContext, 0, cuDevice));
        std::cout << "      OK: CUDA context created" << std::endl;

        // Load NVOF library
        std::cout << std::endl << "[2/7] Loading NVOF library..." << std::endl;
        HMODULE nvofLib = LoadLibraryA("Optical_Flow_SDK_5.0.7\\lib\\x64\\nvofapi64.dll");
        if (!nvofLib) {
            nvofLib = LoadLibraryA("nvofapi64.dll");  // Try system PATH
        }
        if (!nvofLib) {
            throw std::runtime_error("Failed to load nvofapi64.dll");
        }
        std::cout << "      OK: NVOF library loaded" << std::endl;

        // Get API function pointers
        typedef NV_OF_STATUS (NVOFAPI* PFNNvOFAPICreateInstanceCuda)(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST* functionList);
        PFNNvOFAPICreateInstanceCuda NvOFAPICreateInstanceCuda =
            (PFNNvOFAPICreateInstanceCuda)GetProcAddress(nvofLib, "NvOFAPICreateInstanceCuda");

        if (!NvOFAPICreateInstanceCuda) {
            throw std::runtime_error("Failed to get NvOFAPICreateInstanceCuda");
        }

        NV_OF_CUDA_API_FUNCTION_LIST functionList = {};
        CHECK_NVOF(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &functionList));
        std::cout << "      OK: NVOF API instance created" << std::endl;

        // Create NVOF handle
        std::cout << std::endl << "[3/7] Creating NVOF handle..." << std::endl;
        NvOFHandle nvofHandle = nullptr;
        CHECK_NVOF(functionList.nvCreateOpticalFlowCuda(cuContext, &nvofHandle));
        std::cout << "      OK: NVOF handle created" << std::endl;

        // Initialize NVOF
        std::cout << std::endl << "[4/7] Initializing NVOF with 1x1 grid (per-pixel flow)..." << std::endl;
        NV_OF_INIT_PARAMS initParams = {};
        initParams.width = WIDTH;
        initParams.height = HEIGHT;
        initParams.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;  // Per-pixel
        initParams.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        initParams.mode = NV_OF_MODE_OPTICALFLOW;
        initParams.perfLevel = NV_OF_PERF_LEVEL_SLOW;  // Best quality
        initParams.enableExternalHints = NV_OF_FALSE;
        initParams.enableOutputCost = NV_OF_FALSE;
        initParams.hPrivData = nullptr;
        initParams.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
        initParams.enableRoi = NV_OF_FALSE;
        initParams.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
        initParams.enableGlobalFlow = NV_OF_FALSE;
        initParams.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;

        CHECK_NVOF(functionList.nvOFInit(nvofHandle, &initParams));
        std::cout << "      OK: NVOF initialized" << std::endl;

        // Create GPU buffers
        std::cout << std::endl << "[5/7] Creating GPU buffers..." << std::endl;

        NV_OF_BUFFER_DESCRIPTOR inputDesc = {};
        inputDesc.width = WIDTH;
        inputDesc.height = HEIGHT;
        inputDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
        inputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;

        NvOFGPUBufferHandle inputBuffer1 = nullptr, inputBuffer2 = nullptr;
        CHECK_NVOF(functionList.nvOFCreateGPUBufferCuda(nvofHandle, &inputDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &inputBuffer1));
        CHECK_NVOF(functionList.nvOFCreateGPUBufferCuda(nvofHandle, &inputDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &inputBuffer2));

        NV_OF_BUFFER_DESCRIPTOR outputDesc = {};
        outputDesc.width = WIDTH;
        outputDesc.height = HEIGHT;
        outputDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
        outputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;  // Flow vectors

        NvOFGPUBufferHandle outputBuffer = nullptr;
        CHECK_NVOF(functionList.nvOFCreateGPUBufferCuda(nvofHandle, &outputDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &outputBuffer));

        std::cout << "      OK: Created 2 input buffers + 1 output buffer" << std::endl;

        // Benchmark
        std::cout << std::endl << "[6/7] Running benchmark..." << std::endl;
        std::cout << "      Warmup: " << WARMUP_ITERS << " iterations" << std::endl;

        // Setup execute parameters
        NV_OF_EXECUTE_INPUT_PARAMS executeInParams = {};
        executeInParams.inputFrame = inputBuffer1;
        executeInParams.referenceFrame = inputBuffer2;
        executeInParams.externalHints = nullptr;
        executeInParams.disableTemporalHints = NV_OF_FALSE;
        executeInParams.padding = 0;
        executeInParams.hPrivData = nullptr;
        executeInParams.padding2 = 0;
        executeInParams.numRois = 0;
        executeInParams.roiData = nullptr;

        NV_OF_EXECUTE_OUTPUT_PARAMS executeOutParams = {};
        executeOutParams.outputBuffer = outputBuffer;
        executeOutParams.outputCostBuffer = nullptr;
        executeOutParams.hPrivData = nullptr;

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            CHECK_NVOF(functionList.nvOFExecute(nvofHandle, &executeInParams, &executeOutParams));
        }
        CHECK_CUDA(cuCtxSynchronize());

        std::cout << "      Timed run: " << TIMED_ITERS << " iterations" << std::endl;

        // Timed iterations
        std::vector<double> latencies;
        latencies.reserve(TIMED_ITERS);

        for (int i = 0; i < TIMED_ITERS; i++) {
            CHECK_CUDA(cuCtxSynchronize());

            auto start = std::chrono::high_resolution_clock::now();
            CHECK_NVOF(functionList.nvOFExecute(nvofHandle, &executeInParams, &executeOutParams));
            CHECK_CUDA(cuCtxSynchronize());
            auto end = std::chrono::high_resolution_clock::now();

            double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency_ms);
        }

        std::cout << "      OK: Benchmark complete" << std::endl;

        // Compute statistics
        BenchmarkStats stats = computeStats(latencies);

        // Print results
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << " BENCHMARK RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;

        printStats("NVOF Hardware Optical Flow", stats);

        // Comparison with RAFT TensorRT
        const double RAFT_TRT_MEAN = 3.64;  // From benchmark_raft.py results

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << " COMPARISON vs RAFT TensorRT" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
        std::cout << "RAFT TensorRT mean latency: " << RAFT_TRT_MEAN << " ms" << std::endl;
        std::cout << "NVOF mean latency:          " << stats.mean << " ms" << std::endl;
        std::cout << std::endl;

        if (stats.mean < RAFT_TRT_MEAN) {
            double speedup = RAFT_TRT_MEAN / stats.mean;
            std::cout << "RESULT: NVOF is " << speedup << "x FASTER than RAFT TensorRT!" << std::endl;
            std::cout << "        Speedup: " << (RAFT_TRT_MEAN - stats.mean) << " ms saved per inference" << std::endl;
            std::cout << std::endl;
            std::cout << "RECOMMENDATION: Proceed with full NVOF Python integration" << std::endl;
            std::cout << "                This will improve end-to-end watermark removal performance" << std::endl;
        } else {
            double slowdown = stats.mean / RAFT_TRT_MEAN;
            std::cout << "RESULT: NVOF is " << slowdown << "x SLOWER than RAFT TensorRT" << std::endl;
            std::cout << "        Slowdown: " << (stats.mean - RAFT_TRT_MEAN) << " ms lost per inference" << std::endl;
            std::cout << std::endl;
            std::cout << "RECOMMENDATION: Keep current RAFT TensorRT implementation" << std::endl;
            std::cout << "                Do NOT invest time in NVOF Python integration" << std::endl;
            std::cout << "                Focus on RFCNet TensorRT integration instead" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;

        // Cleanup
        functionList.nvOFDestroyGPUBufferCuda(outputBuffer);
        functionList.nvOFDestroyGPUBufferCuda(inputBuffer2);
        functionList.nvOFDestroyGPUBufferCuda(inputBuffer1);
        functionList.nvOFDestroy(nvofHandle);
        FreeLibrary(nvofLib);
        CHECK_CUDA(cuCtxDestroy(cuContext));

        return (stats.mean < RAFT_TRT_MEAN) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << std::endl;
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
