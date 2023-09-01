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
advertising strategies of the merchant. An action is sending a promotion or not. 
Three experiments are implemented.
1. Compare the different implemented estimation models for one period (and using the true customer behaviour)
2. 
