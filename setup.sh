#!/bin/bash

set -e

# Clone submodules
git submodule update --init --recursive

# Create conda environment
conda create -n avhubert_gpu python=3.8 -y
conda activate avhubert_gpu

# Install torch and ffmpeg
conda install pytorch==1.10.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
conda install conda-forge::ffmpeg

# # Install pip and requirements
# python -m pip install --upgrade pip==23.1.2
# pip install -r requirements.txt

# # Install fairseq
# cd modules/avhubert/fairseq
# pip install --editable .
# cd ../../..

# # Install dlib
# cd modules/dlib
# mkdir -p build
# cd build
# cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
# cmake --build .
# cd ..
# python setup.py install --set DLIB_USE_CUDA=1
# cd ../..

# # Download pre-trained models
# ./download.sh
