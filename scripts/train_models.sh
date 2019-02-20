#!/bin/bash
# Trains needed neural networks

python3 -m deep_model_training.cifar10_resnet
python3 -m deep_model_training.norb_resnet
python3 -m deep_model_training.norb_resnet --data_augmentation
