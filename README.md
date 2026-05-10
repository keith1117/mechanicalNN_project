# MulaTOVA: Neural Network-Based Lattice Topology Optimization Interface

MulaTOVA is a PyTorch-based topology optimization framework for AI-assisted mechanical structure design. 

This repository extends the framework with a Flask-based web interface for configuring experiments, running optimization tasks, and visualizing generated results.

## Overview

The underlying framework supports neural network-based lattice topology optimization. My work focused on extending the existing framework with an interactive web interface, experiment controls, and visualization pages.

Through the interface, users can define boundary conditions, define loading conditions, select target shapes, run optimization experiments, and view the generated results.

## Features

- Flask-based web interface for topology optimization experiments
- User-facing controls for defining boundary conditions
- User-facing controls for defining loading conditions
- Target-shape selection for optimization experiments
- Browser-based workflow for running optimization
- Result visualization for:
  - topology outputs
  - displacement maps
  - target fields
  - convergence plots
- Past-result loading from the generated results folder

## My Contributions

- Extended a provided PyTorch topology optimization framework by building a Flask-based interface for AI-assisted mechanical structure design.
- Implemented user-facing controls for defining boundary/loading conditions, selecting target shapes, and running optimization experiments through a web interface.
- Integrated visualization pages for topology outputs, displacement maps, target fields, and convergence plots to support experiment inspection and result interpretation.

## Technologies Used

- Python
- Flask
- PyTorch
- NumPy
- Matplotlib
- YAML configuration files

## Project Structure

```text
.
├── app.py
├── MulaTOVA_MNN.py
├── FE.py
├── material_models.py
├── TO_models.py
├── VAE_Pytorch_Hybrid_to_Hybrid.py
├── utils.py
├── requirements.txt
├── config/
│   └── struct.yaml
├── data/
│   └── struct/
└── results/
    └── struct/
