#!/usr/bin/bash
mkdir -p data/misc
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O data/misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d data/misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O data/misc/20words_mean_face.npy
wget --content-disposition https://dl.fbaipublicfiles.com/avhubert/demo/avhubert_demo_video_8s.mp4 -O data/misc/avhubert_demo_video_8s.mp4
