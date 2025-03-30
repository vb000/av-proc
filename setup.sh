#!/bin/bash

set -e

# Install torch and ffmpeg
conda install -y conda-forge::ffmpeg

# Install pip and requirements
python -m pip install --upgrade pip==23.1.2
pip install -r requirements.txt

# Install fairseq
cd modules/AVHuBERT/fairseq
pip install --editable .
cd ../../..

# Install dlib
cd modules/dlib
mkdir -p build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --set DLIB_USE_CUDA=1
cd ../..

# Download pre-trained models
./download.sh
