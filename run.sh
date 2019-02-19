mkdir dnn_save_path
mkdir dnn_best_model
mkdir dnn_logdir dnn_logdir/train dnn_logdir/test
CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train.py train DIEN  >train_dein2.log 2>&1 &
