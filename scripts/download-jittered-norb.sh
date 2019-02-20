#!/bin/bash
# create directories
data_dir=norb_data
# local directory and source
dir="${data_dir}/jittered/"

mkdir $data_dir
mkdir $dir

source="https://cs.nyu.edu/~ylclab/data/norb-v1.0/"
train="norb-5x46789x9x18x6x2x108x108-training"
test="norb-5x01235x9x18x6x2x108x108-testing"
files=("info" "cat" "dat")
ntrain=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
ntest=("01" "02")

# download and unzip training data
for i in "${ntrain[@]}"
do
	for j in "${files[@]}"
	do
		wget -O "${dir}${train}-${i}-${j}.mat.gz" "${source}${train}-${i}-${j}.mat.gz"
		gunzip "${dir}${train}-${i}-${j}.mat.gz"
	done
done

# download and unzip testing data
for i in "${ntest[@]}"
do
	for j in "${files[@]}"
	do
		wget -O "${dir}${test}-${i}-${j}.mat.gz" "${source}${test}-${i}-${j}.mat.gz"
		gunzip "${dir}${test}-${i}-${j}.mat.gz"
	done
done