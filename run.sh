#!/usr/bin/env bash
model=$1
config=$2
outdir=$3
rm -rf ${outdir}
mkdir ${outdir}
mkdir ${outdir}/best_model
mkdir ${outdir}/logdir ${outdir}/logdir/train ${outdir}/logdir/test
if [ ${model} == "DIEN" ]; then
CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train_dien.py train ${config} ${outdir} > ${outdir}/train_dien.log 2>&1 &
elif [ ${model} == "SEMB" ]; then
CUDA_VISIBLE_DEVICES=0  /usr/bin/python2.7  script/train_semb.py train ${config} ${outdir} > ${outdir}/train_semb.log 2>&1 &
fi
echo "Training started!!"
