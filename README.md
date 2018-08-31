
# ![](documents/dsm.png?raw=true "Icon")

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![python: 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


This Python package is a development framework for demand side management (DSM) research. Cyber-enabled DSM 
plays a crucial role in smart grid by providing automated decision-making capabilities that selectively 
schedule loads on these local grids to improve power balance and grid stability. 

Functions are provided in this package to simulate stochastic load of residential homes using block bootstrap,  
 simulate DSM load profiles of those homes, and also simulate electricity prices by a system operator.

This package was started by Kostas Hatalis in 2018.

## Installation

Clone/fork this repository and then run:
```
python setup.py develop
```

Lehigh-DSM has the following dependencies:
```
pandas
numpy
matplotlib
```