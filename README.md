## Setup

1. Clone

        git clone git@github.com:vb000/av-proc.git
        cd av-proc
        git submodule update --init --recursive

2. Create env

        conda create -n avhubert python=3.8 -y
        conda activate avhubert

3. Downgrade `pip` to 23.1.2

        python -m pip install --upgrade pip==23.1.2

4. Install `fairseq` and `avhubert` dependencies

        pip install -r requirements.txt
        cd modules/avhubert/fairseq
        pip install --editable .

5. Obtain checkpoints

        cd ../../..
        ./download.sh

6. Install `ffmpeg`

        conda install conda-forge::ffmpeg
