## Setup

1. Create env
        conda create -n avhubert python=3.8 -y
        conda activate avhubert
2. Separately, clone the `avhubert` repository.
        git clone https://github.com/facebookresearch/av_hubert.git
        cd avhubert
        git submodule init
        git submodule update
3. Install `fairseq` and `avhubert` dependencies.
        pip install -r requirements.txt
        cd fairseq
        pip install --editable ./
