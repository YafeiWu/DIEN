#!/usr/bin/env bash
model=$1
config=$2
outdir=$3
rm -rf ${outdir}
mkdir ${outdir}
mkdir ${outdir}/best_model
mkdir ${outdir}/logdir ${outdir}/logdir/train ${outdir}/logdir/test
CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train_sess.py train ${config} ${outdir} > ${outdir}/train_sess.log 2>&1 &
echo "Training started!!"
