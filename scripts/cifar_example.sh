#!/bin/bash
# Run tests
python3 LOO_experiments_cifar.py
# Generate plots for tests that completed
python3 gen_all_outputs.py
