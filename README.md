# meta-learners-extension-replication
In this github repository one can find all the code written by Camille Taltas, Francesca Guiso, and Tinatin Nikvashvili to replicate and extend the results of [Metalearners for estimating heterogeneous treatment effects using machine learning](https://www.pnas.org/content/116/10/4156).

In the ```real_data``` folder is the code for all the experiments using the "Get-Out-To-Vote" experiment data:

* ```ml.R``` and ```ml2.R``` run the three meta-learners with honest random forest for figure 2 and 3 respectively. 
* ```real_data_analysis_HRF.ipynb``` creates the plots for figure 2 and 3 with the data from ```ml.R``` and ```ml2.R```
* ```real_data_analysis.ipynb``` includes the code for the three meta-learners with random forest and XGBoost and for creating the plots of figures 2 and 3. 

The ```simulations``` folder includes all the code to run the simulations along with the figures in the simulations part of the paper. The ```.R``` scripts are for the honest random forest simulations and the python scripts for BART, random forest, and XGBoost. 
 
