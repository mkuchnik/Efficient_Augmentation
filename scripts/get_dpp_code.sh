#!/usr/bin/env bash

wget http://www.alexkulesza.com/code/dpp.tgz
tar -xf dpp.tgz
cp dpp/sampling/sample_dpp.m .
cp dpp/helpers/decompose_kernel.m .
cp dpp/helpers/elem_sympoly.m .
cp dpp/helpers/genmult.m .
cp dpp/helpers/sample_k.m .
rm -rf dpp