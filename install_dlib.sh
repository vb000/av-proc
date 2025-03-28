#!/bin/bash
set -e

cd modules/dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DCUDAToolkit_ROOT=$CUDA_BIN
cmake --build .
cd ..
python setup.py install --set DLIB_USE_CUDA=1
cd ../..
