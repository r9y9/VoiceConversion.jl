#!/bin/bash

# Download script for CMU_ARCTIC: http://festvox.org/cmu_arctic/

# The corpus will be downloaded in $HOME/data/cmu_arctic/
location=$HOME/data/cmu_arctic/

if [ ! -e $location ]
then
    echo "Create " $location
    mkdir -p $location
fi

root=http://www.speech.cs.cmu.edu/cmu_arctic/packed/

awb=${root}cmu_us_awb_arctic-0.95-release.zip
bdl=${root}cmu_us_bdl_arctic-0.95-release.zip
clb=${root}cmu_us_clb_arctic-0.95-release.zip
jmk=${root}cmu_us_jmk_arctic-0.95-release.zip
ksp=${root}cmu_us_ksp_arctic-0.95-release.zip
rms=${root}cmu_us_rms_arctic-0.95-release.zip
slt=${root}cmu_us_slt_arctic-0.95-release.zip

cd $location

function download() {
    identifier=$1
    file=$2
    echo "start downloading $identifier"
    mkdir -p tmp
    curl -L -o tmp/${identifier}.zip $file
    unzip tmp/${identifier}.zip
    rm -rf tmp
}

download "awb" $awb
download "bdl" $bdl
download "clb" $clb
download "jmk" $jmk
download "ksp" $ksp
download "rms" $rms
download "slt" $slt
