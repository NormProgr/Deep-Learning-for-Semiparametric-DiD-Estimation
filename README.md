# Deep Learning for Semiparametric DiD Estimation

[![pre-commit.ci passed](https://img.shields.io/badge/pre--commit.ci-passed-brightgreen)](https://results.pre-commit.ci/run/github/274689747/1678058970.SI-lnarDSRqXafVBdLucmg)
[![image](https://img.shields.io/badge/python-3.11.0-blue)](https://www.python.org/)
[![image](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/license/mit/)
[![image](https://img.shields.io/badge/LaTeX-v0.3.0-yellowgreen)](https://www.tug.org/texlive/)

## Abstract

This thesis explores the implementation of deep feedforward neural networks into
semiparametric Difference-in-Differences estimation (DiD), highlighting its potential
under conditional parallel trends assumption. It reviews current classical and deep
learning estimation methods for first-step DiD estimation and conducts a Monte Carlo
Simulation to test their validity for second-step inference. The results demonstrate
that deep learning performs nearly as well as the best classical approaches and
outperforms those in scenarios with incorrectly specified outcomes. To further
investigate deep learning, multiple deep learning architectures are tested, showing
sensitivity towards their hyperparameters. Finally, DiD deep learning estimators show
promise in real-world applications, handling heterogeneous treatment effects
effectively.

## Usage

To get started, create and activate the environment with

```console
$ cd ./deep_learning_for_semiparametric_did_estimation
$ conda env create -f environment.yml
$ conda activate did
```

To create the project, type:

```console
$ pytask
```

this includes creating the pdf of the thesis and running multiple simulations.

## Table of Contents

1. Introduction - 1
1. Methodology - 3
   1. 2x2 Difference in Differences - 3
   1. Outcome Regression - 4
   1. Inverse Probability Weighting - 5
   1. Double Robust Difference in Differences - 6
1. Deep Learning - 8
   1. Revision of Deep Learning - 8
   1. Deep Learning for Inference - 11
1. Monte Carlo Simulations - 13
   1. Data Generating Process - 13
   1. Results Homogenous Treatment Effects - 15
   1. Results Heterogeneous Treatment Effects - 18
   1. Comparison of Deep Learning Architectures - 20
1. Application - 23
1. Discussion and Further Research - 26
1. Conclusion - 27

The papers content is structured in LaTeX in the following directory: */paper/*

## Monte Carlo Simulations

### Classical Methods

Non-deep learning simulations are ordered like that in the paper, which includes the
following estimators:

- Two-Way-Fixed-Effects
  - twfe_dgp1.py
  - twfe_dgp2.py
  - twfe_dgp3.py
  - twfe_dgp4.py
- Inverse Probability Weighting with logit
  - ipw_dgp1.py
  - ipw_dgp2.py
  - ipw_dgp3.py
  - ipw_dgp4.py
- Doubly robust Difference in Differences with logit
  - dr_dgp1.py
  - dr_dgp2.py
  - dr_dgp3.py
  - dr_dgp4.py
- and with heterogenous treatment effects
  - het_twfe_dgp4.py
  - het_ipw_dgp4.py
  - het_dr_dgp4.py

### Deep Learning Methods

Deep learning simulations can be run by running the accompanying jupyter notebook:

- Inverse Probability Weighting + Deep Learning
  - abadie_dl.ipynb
- Doubly Robust Difference in Differences + Deep Learning
  - SZ_dl.ipynb
- Neural Network Architecture with DGP 4 and heterogenous treatment effects
  - architecture_nn.ipynb
- IPW and DR with Deep Learning for DGP 4 and heterogenous treatment effects
  - SZ_dl_heterogenous.ipynb
- Neural Network Architecture with DGP 4 and heterogenous treatment effects
  - architecture_nn_heterogenous.ipynb

to run the code faster it is recommended to use a GPU or run the code in Google Colab.

## Application

The application on the dataset of Meyer, Viscusi, and Durbin (1995) results can be
reproduced by running the following jupyter notebook:

- application.ipynb

## Miscallaneous

For the ReLU activation function visualization, run the following jupyter notebook:

- plot_relu.ipynb

## Code Quality

To check code quality, type

```console
$ pre-commit install
```

## Known Issues

Sometimes when running the LaTeX code, especially using a MikTex distribution, an error
with a missing package called "pearl" appears on windows. To fix this, install the
package manually from here:

https://strawberryperl.com/

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
