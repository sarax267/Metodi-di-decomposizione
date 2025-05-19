#!/bin/bash

# Container per Gauss-Seidel e Gradient Descent (1 CPU), Jacobi (4 CPU)

docker run --rm -v /mnt/c/Users/sarab/OneDrive/Desktop/Unifi\ magistrale/Optimization\ methods/Project\ work\ 3\ cfu/logistic-regression-project:/workspace --cpus="4" logistic-regression python main.py --method gauss_seidel gradient_descent gradient_descent_armijo jacobi
