#!/bin/bash

set -xe

# Install torch and ffmpeg
conda install -y conda-forge::ffmpeg

# Install pip and requirements
python -m pip install --upgrade pip==23.1.2
pip install -r avhubert_requirements.txt

# Install fairseq
cd modules/AVHuBERT/fairseq
pip install --editable .
cd ../../..

# Install dlib
# Thanks to https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781!
conda install -y cuda cudnn -c nvidia
rm -rf modules/dlib/build
cd modules/dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --set DLIB_USE_CUDA=1
cd ../..

# Download pre-trained models
./download.sh
