import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def t_learner(data, x_vars, base_learner):
  # separate outcomes for treated and untreated individuals
  data['Y_0'] = data[data['W']==0]['Y']
  data['Y_1'] = data[data['W']==1]['Y']

  # create separate dataframes for treated and untreated so we can fit base_learner
  treatment_group = data[x_vars+['Y_1']].dropna()
  control_group = data[x_vars+['Y_0']].dropna()

  # Estimate control group response function
  model_0 = base_learner
  # This is fit on the control group
  model_0.fit(control_group[x_vars],control_group['Y_0'])
  mu_0 = model_0.predict_proba(data[x_vars])[:,1]

  # Estimate treatment group response function
  model_1 = base_learner
  # This is fit on the treatment group
  model_1.fit(treatment_group[x_vars],treatment_group['Y_1'])
  mu_1 = model_1.predict_proba(data[x_vars])[:,1]

  cate_t = mu_1 - mu_0

  return cate_t

def s_learner(data, x_vars, base_learner):
  # Estimate combined response function (here, we include W as an input)
  model_combined = base_learner
  model_combined.fit(data[x_vars+['W']],data['Y'])

  # Create dataframe for predictions
  data_w = data[x_vars].copy()

  # Pretend everyone is treated
  # Estimated response when W = 1
  data_w['W=1'] = 1

  # Pretend everyone is control
  # Estimated response when W = 0
  data_w['W=0'] = 0

  # Estimate individual response under assumption that everyone is treated (i.e. W=1)
  mu_1_s = model_combined.predict_proba(data_w[x_vars+['W=1']])[:,1]

  # Estimate individual response under assumption that everyone is in control group (i.e. W=0)
  mu_0_s = model_combined.predict_proba(data_w[x_vars+['W=0']])[:,1]

  cate_s = mu_1_s - mu_0_s

  return cate_s

def x_learner(data, x_vars, base_learner, base_learner_2=LinearRegression()):
  """
  Input:
    data: pd.Dataframe
    base_learner: base learner used to get mu_0, mu_1
    base_learner_2: base learner of the second stage (used to get tau_0, tau_1)
  Output:
    CATE estimate
  """
  # separate outcomes for treated and untreated individuals
  data['Y_0'] = data[data['W']==0]['Y']
  data['Y_1'] = data[data['W']==1]['Y']

  # create separate dataframes for treated and untreated so we can fit base_learner
  treatment_group = data[x_vars+['Y_1']].dropna()
  control_group = data[x_vars+['Y_0']].dropna()

  # Estimate control response function
  # Same as T-learner
  model_0 = base_learner
  model_0.fit(control_group[x_vars],control_group['Y_0'])

  # Use mu_0 to impute potential outcome Y(0) for TREATMENT group
  treatment_group['y_hat_0'] = model_0.predict_proba(treatment_group[x_vars])[:,1]

  # Estimate treatment response function
  # Same as T-learner
  model_1 = base_learner
  model_1.fit(treatment_group[x_vars],treatment_group['Y_1'])

  # Use mu_1 to impute  potential outcome Y(1) for CONTROL group
  control_group['y_hat_1'] = model_1.predict_proba(control_group[x_vars])[:,1]

  # Add treatment flag
  treatment_group['W'] = 1
  control_group['W'] = 0

  imputed_data = pd.concat((treatment_group,control_group))

  # Use imputed potential outcomes to imputed treatment effects
  imputed_data['diff_w=0'] = imputed_data['y_hat_1'] - imputed_data['Y_0'] # y_hat_1 is NaN for W=1
  imputed_data['diff_w=1'] = imputed_data['Y_1'] - imputed_data['y_hat_0'] # y_hat_0 is NaN for W=0

  control_imp = imputed_data[imputed_data['W']==0][x_vars+['diff_w=0']]
  treatment_imp = imputed_data[imputed_data['W']==1][x_vars+['diff_w=1']]

  # Estimate tau_0, tau_1 using "base learners of the second stage"
  # TODO: play around with model used, can be a different model for tau_0,tau_1

  # Fit on imputed data (treatment group)
  control_model = base_learner_2
  control_model.fit(control_imp[x_vars],control_imp['diff_w=0'])
  # tau_0 is a CATE estimate fit to the control group
  tau_0 = control_model.predict(data[x_vars])

  # Fit on imputed data (control group)
  treatment_model = base_learner_2
  treatment_model.fit(treatment_imp[x_vars],treatment_imp['diff_w=1'])
  # tau_1 is a CATE estimate fit to the treatment group
  tau_1 = control_model.predict(data[x_vars])

  # estimate of propensity score
  m_prop = base_learner_2
  m_prop.fit(data[x_vars],data['W'])
  g = m_prop.predict_proba(x_test)[:,1]

  cate_x = g*tau_1 + (1-g)*tau_0

  return cate_x
