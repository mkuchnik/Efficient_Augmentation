#!/bin/bash
# Run tests
echo "Running MNIST experiments"
python3 LOO_experiments.py
echo "Running CIFAR10 experiments"
python3 LOO_experiments_cifar.py
echo "Running NORB experiments"
python3 LOO_experiments_norb.py
# Generate plots for tests that completed
echo "Generating visualized results as PDFs"
python3 gen_all_outputs.py
