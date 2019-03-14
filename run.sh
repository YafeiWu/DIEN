#!/usr/bin/env bash
outdir=$1
config=$2
rm -rf ${outdir}
mkdir ${outdir}
mkdir ${outdir}/dnn_save_path
mkdir ${outdir}/dnn_best_model
mkdir ${outdir}/dnn_logdir ${outdir}/dnn_logdir/train ${outdir}/dnn_logdir/test
CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train_dien.py train ${config} > ${outdir}/train_dein.log 2>&1 &
echo "Training started!!"
