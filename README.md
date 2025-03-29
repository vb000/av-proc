## Setup

1. Clone

        git clone git@github.com:vb000/av-proc.git
        cd av-proc
        git submodule update --init --recursive

2. Create env

        conda create -n avhubert_gpu python=3.8 -y
        conda activate avhubert_gpu
        conda install pytorch==1.10.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
        conda install conda-forge::ffmpeg

3. Downgrade `pip` to 23.1.2

        python -m pip install --upgrade pip==23.1.2

4. Install `fairseq` and `avhubert` dependencies

        pip install -r requirements.txt
        cd modules/avhubert/fairseq
        pip install --editable .
        cd ../../..

5. Install `dlib`

        ./install_dlib.sh

6. Obtain checkpoints

        cd ../../..
        ./download.sh
