"""
Setup script for NVOF CUDA Python extension

Build with:
    python setup_nvof.py build_ext --inplace

This will create nvof_cuda.pyd (Windows) that can be imported in Python
"""

from distutils.core import setup, Extension
import os
from pathlib import Path

# Paths
CUDA_PATH = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
NVOF_SDK_PATH = Path(__file__).parent / "Optical_Flow_SDK_5.0.7"

# Check paths exist
if not CUDA_PATH.exists():
    raise RuntimeError(f"CUDA not found at: {CUDA_PATH}")
if not NVOF_SDK_PATH.exists():
    raise RuntimeError(f"NVOF SDK not found at: {NVOF_SDK_PATH}")

# Extension module configuration
nvof_cuda_ext = Extension(
    'nvof_cuda',
    sources=['nvof_cuda_wrapper.cpp'],
    include_dirs=[
        str(NVOF_SDK_PATH / "NvOFInterface"),
        str(CUDA_PATH / "include"),
    ],
    library_dirs=[
        str(CUDA_PATH / "lib" / "x64"),
    ],
    libraries=[
        'cuda',
        'cudart',
    ],
    extra_compile_args=[
        '/std:c++17',
        '/EHsc',
    ],
    language='c++',
)

setup(
    name='nvof_cuda',
    version='1.0',
    description='NVIDIA Hardware Optical Flow CUDA wrapper',
    ext_modules=[nvof_cuda_ext],
)
