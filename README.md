## Setup

1. Clone

        git clone git@github.com:vb000/av-proc.git
        cd av-proc
        git submodule update --init --recursive

2. Create env

        module load cuda/11.3.0 # Make sure cuda is available
        ./setup.sh
