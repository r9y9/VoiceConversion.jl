#!/bin/bash

# Demonstration script for GMM-based statistical voice conversion. Note that
# this script may take hours to finish.

# Please make sure to download the CMU arctic dataset before running this
# script. It is assumed that the dataset is downloaded in ~/data/cmu_arctic.
# For downloading the dataset, you can use `download_cmu_arctic.sh`.

SCRIPT_DIR="../../scripts"

## Experimental conditions

SRC="clb" # Source speaker identifier
TGT="slt" # Target speaker identifier

DIFF=0    # enable direct waveform modification based on spectrum differencial
MAX=100   # maximum number of training data
ORDER=40  # order of mel-cepstrum (expect for 0th)
MIX=8     # number of mixtures of GMMs
POWER_THRESHOLD=-14 # power threshold to select frames used in training
NITER=30  # number of iteration in EM algorithm

SKIP_FEATURE_EXTRACTION=0
SKIP_FEATURE_ALIGNMENT=0
SKIP_MODEL_TRAINING=0
SKIP_VOICE_CONVERSION=0

## Directory settings

FEATURE_SAVE_DIR="./features"
MODEL_SAVE_DIR="./models"
VC_SAVE_DIR_TOP="./converted_wav"

if [ $DIFF == 1 ]
then
    VC_SAVE_DIR=${VC_SAVE_DIR_TOP}/${SRC}_to_${TGT}_order${ORDER}_gmm${MIX}_diff
else
    VC_SAVE_DIR=${VC_SAVE_DIR_TOP}/${SRC}_to_${TGT}_order${ORDER}_gmm${MIX}
fi

mkdir -p $FEATURE_SAVE_DIR
mkdir -p $MODEL_SAVE_DIR
mkdir -p $VC_SAVE_DIR

## Feature Extraction

if [ $SKIP_FEATURE_EXTRACTION == 0 ]
then
    julia ${SCRIPT_DIR}/mcep.jl \
	~/data/cmu_arctic/cmu_us_${SRC}_arctic/wav/ \
	${FEATURE_SAVE_DIR}/speakers/${SRC}/ --max $MAX --order $ORDER

    julia ${SCRIPT_DIR}/mcep.jl \
	~/data/cmu_arctic/cmu_us_${TGT}_arctic/wav/ \
	${FEATURE_SAVE_DIR}/speakers/${TGT}/ --max $MAX --order $ORDER
fi

## Alignment

if [ $SKIP_FEATURE_ALIGNMENT == 0 ]
then
    julia ${SCRIPT_DIR}/align.jl \
	${FEATURE_SAVE_DIR}/speakers/${SRC} \
	${FEATURE_SAVE_DIR}/speakers/${TGT} \
	${FEATURE_SAVE_DIR}/parallel/${SRC}_and_${TGT}/ \
	--max $MAX \
	--threshold $POWER_THRESHOLD
fi

if [ $SKIP_MODEL_TRAINING == 0 ]
then
    if [ $DIFF == 1 ]
    then
	## Training GMM on joint differencial features

	MODEL_PATH=${MODEL_SAVE_DIR}/${SRC}_to_${TGT}_gmm${MIX}_order${ORDER}_diff.jld

	julia ${SCRIPT_DIR}/train_gmm.jl \
	    ${FEATURE_SAVE_DIR}/parallel/${SRC}_and_${TGT}/ \
	    $MODEL_PATH \
	    --max $MAX \
	    --n_components ${MIX} \
	    --n_iter $NITER \
	    --n_init 1 \
	    --diff
    else
	## Training GMM on joint features

	MODEL_PATH=${MODEL_SAVE_DIR}/${SRC}_and_${TGT}_gmm${MIX}_order${ORDER}.jld

	julia ${SCRIPT_DIR}/train_gmm.jl \
	    ${FEATURE_SAVE_DIR}/parallel/${SRC}_and_${TGT}/ \
	    $MODEL_PATH \
	    --max $MAX \
	    --n_components ${MIX} \
	    --n_iter $NITER \
	    --n_init 1
    fi
fi

## Voice Conversion

if [ $SKIP_VOICE_CONVERSION == 0 ]
then
    if [ $DIFF == 1 ]
    then
	SYNTHESIS_SCRIPT_PATH=${SCRIPT_DIR}/diffvc.jl
    else
	SYNTHESIS_SCRIPT_PATH=${SCRIPT_DIR}/vc.jl
    fi

    for n in `seq -w 100 1 105`
    do
	julia ${SYNTHESIS_SCRIPT_PATH} \
	    ~/data/cmu_arctic/cmu_us_${SRC}_arctic/wav/arctic_a0${n}.wav \
	    $MODEL_PATH \
	    ${VC_SAVE_DIR}/arctic_a0${n}_${SRC}_to_${TGT}.wav \
	    --order $ORDER
    done
fi
