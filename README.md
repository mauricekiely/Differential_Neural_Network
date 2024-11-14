# Differential Neural Network in C++

This project implements a Differential Neural Network (DNN) in C++. It was developed as part of my M.Sc. in High-Performance Computing Final Project. The Differential Neural Network concept is inspired by the work of Antoine Savine, as detailed in [this paper](https://github.com/asavine/differential-ml).

## Overview

The Differential Neural Network is a model designed for high-performance and computationally efficient learning, inspired by techniques in automatic differentiation and gradient-based optimization. The model utilizes a **Twin-Network** architecture, essentially comprising two Neural Networks working in tandem to output a predicted value alongside predicted derivatives of inputs for a given set of inputs.
My implementation showcases this process with randomly generated Black-Scholes Data.

## Features

- **Parallel C++ Implementation**: The code is optimized for high-speed execution and efficient memory usage.
- **Twin-Network Architecture**: Utilizes a dual-network setup to enable differential calculations with enhanced precision.
