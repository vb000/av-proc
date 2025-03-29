#!/bin/bash
# From: https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781

set -e

cd modules/dlib
mkdir -p build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --set DLIB_USE_CUDA=1
cd ../..
