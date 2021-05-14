import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def t_learner(data, x_vars, x_test, base_learner):
    # create separate dataframes for treated and untreated so we can fit base_learner
    control_group = data[data['W'] == 0]
    treatment_group = data[data['W'] == 1]

    # Estimate control group response function
    model_0 = base_learner
    # This is fit on the control group
    model_0.fit(control_group[x_vars],control_group['Y'])
    mu_0 = model_0.predict(x_test)

    # Estimate treatment group response function
    model_1 = base_learner
    # This is fit on the treatment group
    model_1.fit(treatment_group[x_vars],treatment_group['Y'])
    mu_1 = model_1.predict(x_test)

    cate_t = mu_1 - mu_0

    return cate_t

def s_learner(data, x_vars, x_test, base_learner):
    # Estimate combined response function (here, we include W as an input)
    model_combined = base_learner
    model_combined.fit(data[x_vars+['W']],data['Y'])

    # Estimate individual response under assumption that everyone is treated (i.e. W=1)
    x_test['W'] = 1
    mu_1_s = model_combined.predict(x_test)

    # Estimate individual response under assumption that everyone is in control group (i.e. W=0)
    x_test.drop('W', axis = 1, inplace = True)
    x_test['W'] = 0
    mu_0_s = model_combined.predict(x_test)

    cate_s = mu_1_s - mu_0_s
    x_test.drop('W', axis = 1, inplace = True)
    return cate_s

def x_learner(data, x_vars, x_test, base_learner, base_learner_class):
    """
    Input:
    data: pd.Dataframe
    base_learner: base learner used to get mu_0, mu_1
    base_learner_2: base learner of the second stage (used to get tau_0, tau_1)
    Output:
    CATE estimate
    """

    # create separate dataframes for treated and untreated so we can fit base_learner
    control_group = data[data['W'] == 0]
    treatment_group = data[data['W'] == 1]

    # Estimate control response function
    # Same as T-learner
    model_0 = base_learner
    model_0.fit(control_group[x_vars],control_group['Y'])

    # Estimate treatment response function
    # Same as T-learner
    model_1 = base_learner
    model_1.fit(treatment_group[x_vars],treatment_group['Y'])

    r_0 = model_1.predict(control_group[x_vars]) - control_group['Y']
    r_1 = treatment_group['Y'] -  model_0.predict(treatment_group[x_vars])

    m_tau_0 = base_learner
    m_tau_0.fit(control_group[x_vars], r_0)

    m_tau_1 = base_learner
    m_tau_1.fit(treatment_group[x_vars], r_1)

    m_prop = base_learner_class
    m_prop.fit(data[x_vars],data['W'])
    prop_scores = m_prop.predict_proba(x_test)[:,1]

    cate_x = prop_scores * m_tau_0.predict(x_test) + (1-prop_scores) * m_tau_1.predict(x_test)

    return cate_x

def get_results(base_learner, base_learner_class, sim = 'sim1',  n_list =  [2500, 5000, 7500, 10000, 20000, 100000, 200000, 300000]):
    sim_train= pd.read_csv(str(sim) + '_train.csv')
    sim_test = pd.read_csv(str(sim) + '_test.csv')
    results = {'n':[], 't':[], 's':[], 'x':[]}
    x_test = sim_test.drop(['y', 'w', 'tau'], axis = 1)
    if sim == 'sim4':
        x_vars = ['x1', 'x2', 'x3', 'x4', 'x5']
    else:
        x_vars = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
               'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19',
               'x20']
    print('finished loading data')

    for n in n_list:
        results['n'].append(n)
        sim_data = sim_train[:n]
        print(n, sim_data.shape)
        data = pd.DataFrame()
        data['Y'] = sim_data['y'].copy()
        data['W'] = sim_data['w'].copy()
        if sim == 'sim4':
            data[['x1', 'x2', 'x3', 'x4', 'x5']] = sim_data[['x1', 'x2', 'x3', 'x4', 'x5']].copy() # TODO: check if this is the right covariate
        else:
            data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                   'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19',
                   'x20']] = sim_data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                   'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19',
                   'x20']].copy() # TODO: check if this is the right covariate
        print('Cate_t')
        cate_t = t_learner(data, x_vars, x_test, base_learner)
        results['t'].append(np.mean((cate_t-sim_test['tau'])**2))
        print('Cate_s')
        cate_s = s_learner(data, x_vars, x_test, base_learner)
        results['s'].append(np.mean((cate_s-sim_test['tau'])**2))
        print('Cate_x')
        cate_x = x_learner(data,x_vars,x_test,base_learner, base_learner_class)
        results['x'].append(np.mean((cate_x-sim_test['tau'])**2))
    return results

if __name__ == "__main__":
    base_learner_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    base_learner_gb = GradientBoostingRegressor(random_state=42)
    # base_learner_bart = SklearnModel(n_trees = 200, n_burn = 1200, alpha = 0.5, beta = 1)
    gb = GradientBoostingClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    lr = LogisticRegression()
    simulations = ['sim' + str(i) for i in range(1,7)]
    for sim in simulations:
        print('XGBoost', sim)
        results = get_results(base_learner_gb, gb, sim)
        results_pd = pd.DataFrame(results)
        results_pd.to_csv('XGBoost_{}.csv'.format(sim), index = None)
        plt.plot([i/1000 for i in results['n']],results['t'],c='gray',marker='x')
        plt.plot([i/1000 for i in results['n']],results['s'],c='green',marker='x')
        plt.plot([i/1000 for i in results['n']],results['x'],c='blue',marker='x')
        plt.title('Base-learners are XGBoost')
        plt.xlabel('Training size (in 1000)')
        plt.ylabel('MSE')
        plt.legend(labels=['T-learner','S-learner','X-learner'])
        plt.savefig('XGBoost_' + sim)
        plt.close('all')
