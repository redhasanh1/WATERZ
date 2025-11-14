cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set compilers explicitly for Windows (before project() call)
if(WIN32)
    if(NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "C compiler")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "CXX compiler")
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
    endif()
endif()

project(DCNv4TensorRTPlugin LANGUAGES CXX CUDA)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# CUDA Configuration
# ============================================================================

find_package(CUDAToolkit REQUIRED)

# RTX 4090 Ada Lovelace: Compute Capability 8.9
# Also add 7.5 (Turing) and 8.0 (Ampere) for compatibility
set(CMAKE_CUDA_ARCHITECTURES 75 80 89)

# CUDA compiler flags
if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# TensorRT Configuration
# ============================================================================

# User can specify TensorRT path via -DTENSORRT_DIR=...
if(NOT DEFINED TENSORRT_DIR)
    if(WIN32)
        # Default TensorRT path on Windows
        set(TENSORRT_DIR "D:/watermarkz/TensorRT-10.13.3.9" CACHE PATH "TensorRT directory")
    else()
        set(TENSORRT_DIR "/usr/local/TensorRT-10.13.3.9" CACHE PATH "TensorRT directory")
    endif()
endif()

message(STATUS "TensorRT directory: ${TENSORRT_DIR}")

# Find TensorRT headers
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS ${TENSORRT_DIR}
    PATH_SUFFIXES include
    REQUIRED)

# Find TensorRT libraries (TensorRT 10.x uses versioned library names)
find_library(TENSORRT_NVINFER
    NAMES nvinfer_10 nvinfer
    PATHS ${TENSORRT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    REQUIRED)

find_library(TENSORRT_NVINFER_PLUGIN
    NAMES nvinfer_plugin_10 nvinfer_plugin
    PATHS ${TENSORRT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
    REQUIRED)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "TensorRT plugin: ${TENSORRT_NVINFER_PLUGIN}")

# ============================================================================
# Source Files
# ============================================================================

set(PLUGIN_SOURCES
    src/DCNv4Plugin.cpp
    src/DCNv4PluginKernel.cu
)

set(PLUGIN_HEADERS
    include/DCNv4Plugin.h
    src/common.h
)

# ============================================================================
# Build Shared Library
# ============================================================================

add_library(dcnv4_plugin SHARED ${PLUGIN_SOURCES} ${PLUGIN_HEADERS})

# Include directories
target_include_directories(dcnv4_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(dcnv4_plugin PRIVATE
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVINFER_PLUGIN}
    CUDA::cudart
    CUDA::cuda_driver
)

# Windows-specific settings
if(WIN32)
    # Export all symbols for Windows DLL
    set_target_properties(dcnv4_plugin PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
        # Disable PDB directory flag for CUDA files (causes nvcc issues)
        CUDA_USE_STATIC_CUDA_RUNTIME OFF
    )

    # Add /MD flag for CXX files only (not CUDA)
    target_compile_options(dcnv4_plugin PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/MD>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/MDd>
    )

    # CUDA files: use simpler flags
    target_compile_options(dcnv4_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>
    )
endif()

# Linux-specific settings
if(UNIX)
    set_target_properties(dcnv4_plugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        POSITION_INDEPENDENT_CODE ON
    )

    # Add -fPIC flag
    target_compile_options(dcnv4_plugin PRIVATE -fPIC)
endif()

# CUDA-specific compilation flags
# Note: CUDA separable compilation causes issues with MSVC on Windows
if(UNIX)
    set_property(TARGET dcnv4_plugin PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET dcnv4_plugin PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# ============================================================================
# Installation (Optional)
# ============================================================================

install(TARGETS dcnv4_plugin
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/DCNv4Plugin.h
    DESTINATION include
)

# ============================================================================
# Build Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "DCNv4 TensorRT Plugin Configuration")
message(STATUS "========================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version:      ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT dir:      ${TENSORRT_DIR}")
message(STATUS "TensorRT include:  ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Output library:    dcnv4_plugin")
if(WIN32)
    message(STATUS "Output file:       dcnv4_plugin.dll")
else()
    message(STATUS "Output file:       libdcnv4_plugin.so")
endif()
message(STATUS "========================================")
message(STATUS "")
