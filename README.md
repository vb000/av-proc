## Setup

1. Clone

        git clone git@github.com:vb000/av-proc.git
        cd av-proc
        git submodule update --init --recursive

2. Create env

        conda create -n avhubert_gpu python=3.8 -y
        conda activate avhubert_gpu

3. Install dependencies

        ./setup.sh
