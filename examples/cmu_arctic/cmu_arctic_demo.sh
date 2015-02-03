#!/bin/bash

# Demonstration script for statistical voice conversion based on spectrum
# differencial. Note that this script may take hours to finish.

# Please make sure to download the CMU arctic dataset before running this
# script. It is assumed that the dataset is downloaded in ~/data/cmu_arctic.
# For downloading the dataset, you can use scripts/download_cmu_arctic.sh.

script_dir="../../scripts"

feature_save_dir="./features"
model_save_dir="./models"
vc_save_dir="./converted_wav"

mkdir -p $feature_save_dir
mkdir -p $model_save_dir
mkdir -p $vc_save_dir

SRC="clb" # Source speaker identifier
TGT="slt" # Target speaker identifier

MAX=100   # maximum number of training data
ORDER=40  # order of mel-cepstrum (expect for 0th)
MIX=8     # number of mixtures of GMMs
POWER_THRESHOLD=-14 # power threshold to select frames used in training
NITER=30  # number of iteration in EM algorithm

## Feature Extraction

julia ${script_dir}/mcep.jl \
    ~/data/cmu_arctic/cmu_us_${SRC}_arctic/wav/ \
    ${feature_save_dir}/speakers/${SRC}/ --max $MAX --order $ORDER

julia ${script_dir}/mcep.jl \
    ~/data/cmu_arctic/cmu_us_${TGT}_arctic/wav/ \
    ${feature_save_dir}/speakers/${TGT}/ --max $MAX --order $ORDER

## Alignment

julia ${script_dir}/align.jl \
    ${feature_save_dir}/speakers/${SRC} \
    ${feature_save_dir}/speakers/${TGT} \
    ${feature_save_dir}/parallel/${SRC}_and_${TGT}/ \
    --max $MAX \
    --threshold $POWER_THRESHOLD

## Training differencial GMM

model_path=${model_save_dir}/${SRC}_to_${TGT}_gmm${MIX}_order${ORDER}_max${MAX}_diff.jld

julia ${script_dir}/train_gmm.jl \
    ${feature_save_dir}/parallel/${SRC}_and_${TGT}/ \
    $model_path \
    --max $MAX \
    --n_components ${MIX} \
    --n_iter $NITER \
    --n_init 1 \
    --diff

## Diff VC

for n in `seq -w 100 1 105`
do
    julia ${script_dir}/diffvc.jl \
	~/data/cmu_arctic/cmu_us_${SRC}_arctic/wav/arctic_a0${n}.wav \
	$model_path \
	${vc_save_dir}/arctic_a0${n}_${SRC}_to_${TGT}_diff.wav \
	--order $ORDER
done
