#!/usr/bin/env bash
task=$1
config=$2
outdir=$3
if [ ${task} == "train" ];then
    rm -rf ${outdir}
    mkdir ${outdir}
    mkdir ${outdir}/best_model
    mkdir ${outdir}/logdir ${outdir}/logdir/train ${outdir}/logdir/test
fi
setsid python script/train_sess.py ${task} ${config} ${outdir} > ${outdir}/${task}_sess.log 2>&1 &
echo "${task} started!! \t" date
