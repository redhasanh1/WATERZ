/**
 * NVOF CUDA Wrapper - Python C Extension
 *
 * Exposes NVIDIA Hardware Optical Flow API to Python/PyTorch
 *
 * Purpose: Replace RAFT optical flow with hardware-accelerated NVOF for RTX 5070
 * Expected performance: Faster than RAFT TensorRT (currently 3.64ms)
 *
 * Architecture:
 *   Python → nvof_adapter.py → this C extension → NVOF SDK → GPU hardware
 *
 * Build:
 *   python setup_nvof.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda.h>
#include <cuda_runtime.h>

// NVOF SDK headers
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

#include <cstring>
#include <stdexcept>

// Global NVOF API function list (loaded once)
static NV_OF_CUDA_API_FUNCTION_LIST g_nvofAPI = {};
static bool g_apiInitialized = false;

// Error handling macro
#define CHECK_NVOF(call) \
    do { \
        NV_OF_STATUS status = (call); \
        if (status != NV_OF_SUCCESS) { \
            PyErr_Format(PyExc_RuntimeError, "NVOF error at %s:%d - status %d", __FILE__, __LINE__, status); \
            return NULL; \
        } \
    } while(0)

#define CHECK_CUDA(call) \
    do { \
        CUresult result = (call); \
        if (result != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(result, &errStr); \
            PyErr_Format(PyExc_RuntimeError, "CUDA error at %s:%d - %s", __FILE__, __LINE__, errStr ? errStr : "Unknown"); \
            return NULL; \
        } \
    } while(0)

/**
 * Initialize NVOF API (call once at module load)
 * Returns: None
 */
static PyObject* nvof_init_api(PyObject* self, PyObject* args) {
    if (g_apiInitialized) {
        Py_RETURN_NONE;
    }

    // Initialize CUDA
    CHECK_CUDA(cuInit(0));

    // Load NVOF library and get API function pointers
    NV_OF_STATUS status = NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &g_nvofAPI);
    if (status != NV_OF_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError,
            "Failed to initialize NVOF API. Status: %d. "
            "Make sure nvofapi64.dll is in PATH or SDK lib/x64/ folder.",
            status);
        return NULL;
    }

    g_apiInitialized = true;
    Py_RETURN_NONE;
}

/**
 * Create NVOF handle for a CUDA device
 * Args: device_id (int) - CUDA device index (default 0)
 * Returns: handle (PyCapsule) - Opaque NVOF handle
 */
static PyObject* nvof_create_handle(PyObject* self, PyObject* args) {
    int deviceId = 0;
    if (!PyArg_ParseTuple(args, "|i", &deviceId)) {
        return NULL;
    }

    if (!g_apiInitialized) {
        PyErr_SetString(PyExc_RuntimeError, "NVOF API not initialized. Call nvof_init_api() first.");
        return NULL;
    }

    // Get CUDA device and create context
    CUdevice cuDevice;
    CHECK_CUDA(cuDeviceGet(&cuDevice, deviceId));

    CUcontext cuContext;
    CHECK_CUDA(cuCtxCreate(&cuContext, 0, cuDevice));

    // Create NVOF handle
    NvOFHandle nvofHandle = nullptr;
    CHECK_NVOF(g_nvofAPI.nvCreateOpticalFlowCuda(cuContext, &nvofHandle));

    if (!nvofHandle) {
        cuCtxDestroy(cuContext);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NVOF handle - returned NULL");
        return NULL;
    }

    // Package handle and context into a capsule
    struct NVOFContext {
        NvOFHandle handle;
        CUcontext context;
    };

    NVOFContext* ctx = new NVOFContext{nvofHandle, cuContext};

    // Create Python capsule with destructor
    auto destructor = [](PyObject* capsule) {
        NVOFContext* ctx = static_cast<NVOFContext*>(PyCapsule_GetPointer(capsule, "nvof_handle"));
        if (ctx) {
            if (g_apiInitialized && ctx->handle) {
                g_nvofAPI.nvOFDestroy(ctx->handle);
            }
            if (ctx->context) {
                cuCtxDestroy(ctx->context);
            }
            delete ctx;
        }
    };

    return PyCapsule_New(ctx, "nvof_handle", destructor);
}

/**
 * Initialize NVOF session with parameters
 * Args:
 *   handle (PyCapsule) - NVOF handle from nvof_create_handle()
 *   width (int) - Frame width
 *   height (int) - Frame height
 *   grid_size (int) - Output grid size (1, 2, or 4)
 *   enable_bidirectional (bool) - Enable backward flow
 * Returns: None
 */
static PyObject* nvof_initialize(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* handleCapsule = nullptr;
    int width, height, gridSize;
    int enableBidirectional = 0;

    static char* kwlist[] = {
        (char*)"handle", (char*)"width", (char*)"height",
        (char*)"grid_size", (char*)"enable_bidirectional", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiii|p", kwlist,
                                     &handleCapsule, &width, &height, &gridSize, &enableBidirectional)) {
        return NULL;
    }

    NVOFContext* ctx = static_cast<NVOFContext*>(PyCapsule_GetPointer(handleCapsule, "nvof_handle"));
    if (!ctx || !ctx->handle) {
        PyErr_SetString(PyExc_ValueError, "Invalid NVOF handle");
        return NULL;
    }

    // Convert grid size to enum
    NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSizeEnum;
    switch (gridSize) {
        case 1: gridSizeEnum = NV_OF_OUTPUT_VECTOR_GRID_SIZE_1; break;
        case 2: gridSizeEnum = NV_OF_OUTPUT_VECTOR_GRID_SIZE_2; break;
        case 4: gridSizeEnum = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4; break;
        default:
            PyErr_Format(PyExc_ValueError, "Invalid grid size %d. Must be 1, 2, or 4.", gridSize);
            return NULL;
    }

    // Setup initialization parameters
    NV_OF_INIT_PARAMS initParams = {};
    initParams.width = width;
    initParams.height = height;
    initParams.outGridSize = gridSizeEnum;
    initParams.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    initParams.mode = NV_OF_MODE_OPTICALFLOW;
    initParams.perfLevel = NV_OF_PERF_LEVEL_SLOW;  // Best quality
    initParams.enableExternalHints = NV_OF_FALSE;
    initParams.enableOutputCost = NV_OF_FALSE;
    initParams.hPrivData = nullptr;
    initParams.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    initParams.enableRoi = NV_OF_FALSE;
    initParams.predDirection = enableBidirectional ? NV_OF_PRED_DIRECTION_BOTH : NV_OF_PRED_DIRECTION_FORWARD;
    initParams.enableGlobalFlow = NV_OF_FALSE;
    initParams.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;  // Will convert from RGB

    CHECK_NVOF(g_nvofAPI.nvOFInit(ctx->handle, &initParams));

    Py_RETURN_NONE;
}

/**
 * Execute optical flow computation
 * Args:
 *   handle (PyCapsule) - NVOF handle
 *   input_ptr (int) - CUDA device pointer to input frame (float32 CHW, normalized 0-1)
 *   reference_ptr (int) - CUDA device pointer to reference frame (float32 CHW, normalized 0-1)
 *   flow_fwd_ptr (int) - CUDA device pointer to output forward flow (float32, 2xHxW)
 *   flow_bwd_ptr (int) - CUDA device pointer to output backward flow (float32, 2xHxW) or 0 if not needed
 *   width (int) - Frame width
 *   height (int) - Frame height
 * Returns: None
 *
 * NOTE: This is a simplified version that assumes pre-allocated buffers.
 *       Full version would handle NVOF buffer creation/management internally.
 */
static PyObject* nvof_execute(PyObject* self, PyObject* args) {
    PyObject* handleCapsule = nullptr;
    unsigned long long inputPtr, referencePtr, flowFwdPtr, flowBwdPtr;
    int width, height;

    if (!PyArg_ParseTuple(args, "OKKKKii",
                          &handleCapsule, &inputPtr, &referencePtr,
                          &flowFwdPtr, &flowBwdPtr, &width, &height)) {
        return NULL;
    }

    NVOFContext* ctx = static_cast<NVOFContext*>(PyCapsule_GetPointer(handleCapsule, "nvof_handle"));
    if (!ctx || !ctx->handle) {
        PyErr_SetString(PyExc_ValueError, "Invalid NVOF handle");
        return NULL;
    }

    // TODO: Implement actual NVOF execution
    // This requires:
    // 1. Creating NVOF GPU buffers from CUDA device pointers
    // 2. Converting PyTorch RGB float tensors to NVOF ABGR8 format
    // 3. Executing NVOF
    // 4. Converting NVOF flow vectors (S10.5 format) back to float32
    // 5. Copying results to output buffers

    PyErr_SetString(PyExc_NotImplementedError,
        "nvof_execute() not yet implemented. Requires buffer conversion logic.");
    return NULL;
}

/**
 * Get NVOF capabilities
 * Args: handle (PyCapsule) - NVOF handle
 * Returns: dict with capabilities (grid_sizes, width_min, width_max, height_min, height_max)
 */
static PyObject* nvof_get_capabilities(PyObject* self, PyObject* args) {
    PyObject* handleCapsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &handleCapsule)) {
        return NULL;
    }

    NVOFContext* ctx = static_cast<NVOFContext*>(PyCapsule_GetPointer(handleCapsule, "nvof_handle"));
    if (!ctx || !ctx->handle) {
        PyErr_SetString(PyExc_ValueError, "Invalid NVOF handle");
        return NULL;
    }

    PyObject* capsDict = PyDict_New();

    // Query capabilities
    struct CapQuery {
        NV_OF_CAPS param;
        const char* name;
    };

    CapQuery queries[] = {
        {NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, "grid_sizes"},
        {NV_OF_CAPS_WIDTH_MIN, "width_min"},
        {NV_OF_CAPS_WIDTH_MAX, "width_max"},
        {NV_OF_CAPS_HEIGHT_MIN, "height_min"},
        {NV_OF_CAPS_HEIGHT_MAX, "height_max"},
    };

    for (const auto& query : queries) {
        uint32_t value = 0;
        uint32_t size = 1;
        NV_OF_STATUS status = g_nvofAPI.nvOFGetCaps(ctx->handle, query.param, &value, &size);
        if (status == NV_OF_SUCCESS) {
            PyDict_SetItemString(capsDict, query.name, PyLong_FromUnsignedLong(value));
        }
    }

    return capsDict;
}

// Module method definitions
static PyMethodDef NVOFMethods[] = {
    {"init_api", nvof_init_api, METH_NOARGS,
     "Initialize NVOF API (call once at startup)"},

    {"create_handle", nvof_create_handle, METH_VARARGS,
     "Create NVOF handle for a CUDA device. Args: device_id (int, default 0). Returns: handle capsule"},

    {"initialize", (PyCFunction)nvof_initialize, METH_VARARGS | METH_KEYWORDS,
     "Initialize NVOF session. Args: handle, width, height, grid_size, enable_bidirectional"},

    {"execute", nvof_execute, METH_VARARGS,
     "Execute optical flow computation (NOT YET IMPLEMENTED)"},

    {"get_capabilities", nvof_get_capabilities, METH_VARARGS,
     "Get NVOF hardware capabilities. Args: handle. Returns: dict"},

    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef nvofmodule = {
    PyModuleDef_HEAD_INIT,
    "nvof_cuda",
    "NVIDIA Hardware Optical Flow CUDA wrapper for Python/PyTorch",
    -1,
    NVOFMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_nvof_cuda(void) {
    return PyModule_Create(&nvofmodule);
}
