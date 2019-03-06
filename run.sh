rm -rf output
mkdir output
mkdir output/dnn_save_path
mkdir output/dnn_best_model
mkdir output/dnn_logdir output/dnn_logdir/train output/dnn_logdir/test
CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train.py train script/config.yml  >output/train_dein2.log 2>&1 &
