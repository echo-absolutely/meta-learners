# meta-learners-extension-replication
This repository contains code written by Camille Taltas, Francesca Guiso, and Tinatin Nikvashvili to replicate and extend the results of [Metalearners for estimating heterogeneous treatment effects using machine learning](https://www.pnas.org/content/116/10/4156).

The ```real_data``` folder contains code used to replicate and extend all the experiments using the "Get-Out-To-Vote" experiment data:

* ```ml.R``` and ```ml2.R``` run the three meta-learners with honest random forest for figure 2 and 3 respectively. 
* ```real_data_analysis_HRF.ipynb``` creates the plots for figure 2 and 3 with the data from ```ml.R``` and ```ml2.R```
* ```real_data_analysis.ipynb``` includes the code for the three meta-learners with random forest and XGBoost and for creating the plots of figures 2 and 3. 

The ```simulations``` folder contains all the code used to run the simulations and to produce results based on this data. 
* ```.R``` scripts are used to reproduce the simulation results using Honest Random Forest as the base-learner 
* ```.py``` scripts use BART, Random Forest, and XGBoost as base-learners.
