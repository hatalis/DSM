
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

## Example Simulation
```python
import lehighdsm.functions as dsm
import numpy as np

# simulation parameters
experiment = {}
experiment['filename'] = 'data/home_all.csv'
experiment['N'] = 200 # number of homes to simulate
experiment['alpha'] = 40_000 # max load tolerated by SO
experiment['beta'] = 1_000 # price for max load
experiment['omega'] = 100 # max load allowed by DSM
experiment['epsilon_D'] = -0.6 # elasticity of load
experiment['epsilon_P'] = -0.4 # elasticity of price
experiment['T'] = 24*7 # total time to simulate

# % of load of each home participating in DSM
experiment['kappa'] = 0 # no DSM

# run simulation
experiment = dsm.load(experiment) # load template homes
experiment = dsm.simulate_city(experiment) # simulate phi
experiment = dsm.simulate_load_price(experiment)
dsm.output_results(experiment)
```
This would give the following results:
# ![](documents/nodsm.png?raw=true "Icon")


