# direct-marketing
Data-Driven Direct Marketing Strategies for a Heterogeneous Consumer Base with History-Dependent Demand

The README.md provides an overview for this repository used during our research project.

The main part of our code is implemented in python in different jupyter notebooks (.ipynb) and seperat python files (.py). 

## Jupyter Notebooks 
### models.ipynb 
The scripts in models.ipynb uses functions implemented in functions.py.
models.ipynb provides scripts to create
1. the customer behaviour
1. the training data for the predictors for customer behaviour based on the customers history 
1. different models used for estimation
1. the evaluation of the estimators with a test data

### simulation.ipynb 
The scripts in simulations.ipynb uses functions implemented in functions.py and dp.py.
simulation.ipynb provides scripts to simulate the customer behaviour with different 
advertising strategies of the merchant. An action is sending a promotion or not. Also integrates the customer value calculated in ampl.

Three experiments are implemented:
1. Compare different estimation models over one period (and compare with the true customer behaviour)
1. Compare different customer selection policies over one period 
1. Compare different customer selection policies over 24 periods

### describe_cust_behaviour.ipynb
describe_cust_behaviour.ipynb provides descriptive statistics for the created customer behaviour. 

### dynamic_opt.ipynb
The algorithm implemented in dynamic_opt.ipynb varies the costs in the dynamic program so that exactly as many customers have a positive customer value according to the advertising budget.

## Additional Functions
### functions.py
The python file provides functions used in the jupyter notebooks. The functions create customer behaviour, build and evaluate estimation models, implement customer selection policies (e. g. linear programm for short term profit maximization)

### dp.py
Calculates the expected customer value over an infite horizon. Implements a dynamic programm with recursion in python. 

## AMPL 
### calculate the expected customer value 
In ampl/DP_Kundenauswahl.txt the expected customer value over an infinte horizon is calculated. To use the provided script "ampl" and the solver "minos" is requiured. The ampl implementation is much faster as the python implementation.

## Other Folders
The other folders provides measures and figures of the evaluation created with the jupyter notebooks.


