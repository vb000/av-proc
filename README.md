## Setup

1. Clone

        git clone git@github.com:vb000/av-proc.git
        cd av-proc
        git submodule update --init --recursive

2. Env for AV-HuBERT inference

        conda create -n avhubert_gpu python=3.8 -y
        conda activate avhubert_gpu
        ./avhubert_env_setup.sh

3. Env for Whisper inference using [whisperX](https://github.com/m-bain/whisperX/tree/v3.3.1)

        conda create -n whisperx python=3.10
        conda activate whisperx
        conda install -y conda-forge::ffmpeg
        conda install -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
        pip install -r whisperx_requirements.txt
