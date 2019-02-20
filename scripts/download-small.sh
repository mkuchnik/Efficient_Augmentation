#!/bin/bash
# create directories
data_dir=norb_data
# local directory and source
dir="${data_dir}/small-norb/"

mkdir $data_dir
mkdir $dir

source="https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
train="smallnorb-5x46789x9x18x6x2x96x96-training"
test="smallnorb-5x01235x9x18x6x2x96x96-testing"
files=("info" "cat" "dat")

# download and unzip training data
for i in "${files[@]}"
do
	wget -O "${dir}${train}-${i}.mat.gz" "${source}${train}-${i}.mat.gz"
	gunzip "${dir}${train}-${i}.mat.gz"
done

# download and unzip testing data
for i in "${files[@]}"
do
	wget -O "${dir}${test}-${i}.mat.gz" "${source}${test}-${i}.mat.gz"
	gunzip "${dir}${test}-${i}.mat.gz"
done