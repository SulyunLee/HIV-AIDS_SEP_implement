from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd

def calculate_vif(X, thresh=0.5):
    dropped = True
    while dropped:
        variables = X.columns
        dropped = False
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped = True
        return X

def linreg_model(feature_name, label_name, df, write_filename):
    # X = calculate_vif(X)
    # formula = '{}~{}'.format(label_name, '+'.join(feature_name))
    X = df[feature_name]
    y = df[label_name]

    model = sm.OLS(y,X, missing='drop').fit()

    # write to csv file
    with open(write_filename, 'w') as fh:
        fh.write(model.summary().as_csv())

    return model

def model_rf(train_X, train_y, test_X, test_y, n_estimators):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    test_r2 = r2_score(test_y, y_pred)
    return test_r2


def rf_model_find_param(feature_name, label_name, df):
    X = df[feature_name]
    y = df[label_name]

    n_estimator_lst = [2, 4, 8, 16, 32]
    results = []
    kf = KFold(n_splits=5)
    for n_estimator in n_estimator_lst:
        score_sum = 0
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            test_r2 = model_rf(X_train, y_train, X_test, y_test, n_estimator)
            score_sum += test_r2
        results.append(score_sum / 5)

    best_param = n_estimator_lst[results.index(max(results))]

    return best_param, max(results)

def rf_simulate(best_param, model_feature, model_label, df, changing_var_name, changing_value):
    model = RandomForestRegressor(n_estimators = best_param)
    X = df[model_feature]
    y = df[model_label]
    model.fit(X, y)
    
    df_changed = df.copy()
    df_changed[changing_var_name] = changing_value
    X_changed = df_changed[model_feature]

    y_pred_simulated = model.predict(X_changed)
    df_changed[model_label] = y_pred_simulated

    result_df = df_changed.groupby('Year')[model_label].sum()

    return result_df

def linreg_simulate(model_feature, model_label, df, changing_var_name, changing_value):
    model = LinearRegression()
    X = df[model_feature]
    y = df[model_label]
    model.fit(X, y)

    df_changed = df.copy()
    df_changed[changing_var_name] = changing_value
    X_changed = df_changed[model_feature]

    y_pred_simulated = model.predict(X_changed)
    df_changed[model_label] = y_pred_simulated

    result_df = df_changed.groupby('Year')[model_label].sum()

    return result_df

def linreg_simulate_num_SSPs(model_feature, model_label, df):
    # change the number of SSPs to 0
    simulation_result1 = linreg_simulate(model_feature, model_label, df, "NUMBER OF SSPs", 0)

    # change the number of SSPs to half
    simulation_result2 = linreg_simulate(model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']/2)

    # change the number of SSPs to double
    simulation_result3 = linreg_simulate(model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']*2)

    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2, simulation_result3], axis=1)
    result_df.columns = ['Actual', 'Simulated zero', 'Simulated half', 'Simulated double']

    result_df.to_csv("results/linreg_{}_numSSPs_simulation_result.csv".format(model_label))

def linreg_iowa_simulate_num_SSPs(model_feature, model_label, df, changed_SSPs, model):
    changed_df = df.copy()
    changed_df["NUMBER OF SSPs"] = changed_SSPs

    X = changed_df[model_feature]
    y_pred_simulated = model.predict(X)
    
    changed_df[model_label] = y_pred_simulated
    
    return y_pred_simulated
    

def linreg_simulate_SSP_legality(model_feature, model_label, df):
    # change the SSP legality to 0
    simulation_result1 = linreg_simulate(model_feature, model_label, df, "SSP LEGALITY BINARY", 0)
    
    # change the SSP legality to 1
    simulation_result2 = linreg_simulate(model_feature, model_label, df, "SSP LEGALITY BINARY", 1)

    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2], axis=1)
    result_df.columns = ['Actual', 'All SSP illegal', "All SSP legal"]

    result_df.to_csv("results/linreg_{}_SSPLegality_simulation_result.csv".format(model_label))

def rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df):
    # change the number of SSPs to 0
    simulation_result1 = rf_simulate(rf_best_param, model_feature, model_label, df, "NUMBER OF SSPs", 0)

    # change the number of SSPs to half
    simulation_result2 = rf_simulate(rf_best_param, model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']/2)

    # change the number of SSPs to double
    simulation_result3 = rf_simulate(rf_best_param, model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']*2)

    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2, simulation_result3], axis=1)
    result_df.columns = ['Actual', 'Simulated zero', 'Simulated half', 'Simulated double']

    result_df.to_csv("results/rf_{}_numSSPs_simulation_result.csv".format(model_label))

def rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df):
    # change the SSP legality to 0
    simulation_result1 = rf_simulate(rf_best_param, model_feature, model_label, df, "SSP LEGALITY BINARY", 0)

    # change the SSP legality to 1
    simulation_result2 = rf_simulate(rf_best_param, model_feature, model_label, df, "SSP LEGALITY BINARY", 1)
    
    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2], axis=1)
    result_df.columns = ['Actual', 'All SSP illegal', 'All SSP legal']

    result_df.to_csv("results/rf_{}_SSPLegality_simulation_result.csv".format(model_label))

if __name__ == "__main__":
    data_dir = "cleaned_dataset/"
    results_dir = "results/"

    dataset_filename = data_dir + "cleaned_final_dataset.csv"
    df = pd.read_csv(dataset_filename)

    # model_feature_withcontrol = ["RW_client", "NUMBER OF SSPs", "SSP LEGALITY BINARY", "No HS diploma", "Poverty", "Uninsured"]
    model_feature = ["NUMBER OF SSPs", "SSP LEGALITY BINARY", "American Indian/Alaska Native rate", "Asian rate", "Black/African American rate", 
                    "Native Hawaiian/Other Pacific Islander rate", "White rate", "No HS diploma percent", "Poverty percent", "Uninsured percent",
                    "Syphilis rate per 100000", "RW client per 100000"]
    df = df.dropna(subset=['Syphilis rate per 100000']).reset_index()

    iowa_df = df[df['Geography'] == 'Iowa']
    iowa_future_df = pd.read_csv("{}current_iowa.csv".format(data_dir))

    conn_SSP = df.loc[df['Geography'] == 'Connecticut','NUMBER OF SSPs'].reset_index()
    arkan_SSP = df.loc[df['Geography'] == 'Arkansas', 'NUMBER OF SSPs'].reset_index()

    ################# predict HIV diagnoses #######################
    model_label = "HIV diagnoses per 100000"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df, "{}{}_linreg_model.csv".format(results_dir, model_label))
    conn_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, conn_SSP, linreg_result)
    arkan_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, arkan_SSP, linreg_result)
    state_SSP = pd.concat([conn_result,arkan_result], axis=1)
    state_SSP.columns = [model_label, 'Arkansas simulation']
    state_SSP['Year'] = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    state_SSP.to_csv("linreg_state_simulation_{}.csv".format(model_label), index=False)

    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df)
    linreg_simulate_SSP_legality(model_feature, model_label, df)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df)

    ################# predict HIV deaths ###########################3
    model_label = "HIV deaths per 100000"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df, "{}{}_linreg_model.csv".format(results_dir, model_label))

    conn_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, conn_SSP, linreg_result)
    arkan_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, arkan_SSP, linreg_result)
    state_SSP = pd.concat([conn_result,arkan_result], axis=1)
    state_SSP.columns = [model_label, 'Arkansas simulation']
    state_SSP['Year'] = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    state_SSP.to_csv("linreg_state_simulation_{}.csv".format(model_label), index=False)

    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df)
    linreg_simulate_SSP_legality(model_feature, model_label, df)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df)

    ################## predict AIDS diagnoses ##########################
    model_label = "AIDS diagnoses per 100000"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df, "{}{}_linreg_model.csv".format(results_dir, model_label))

    conn_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, conn_SSP, linreg_result)
    arkan_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, arkan_SSP, linreg_result)
    state_SSP = pd.concat([conn_result,arkan_result], axis=1)
    state_SSP.columns = [model_label, 'Arkansas simulation']
    state_SSP['Year'] = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    state_SSP.to_csv("linreg_state_simulation_{}.csv".format(model_label), index=False)
    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df)
    linreg_simulate_SSP_legality(model_feature, model_label, df)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df)

    #################### predict AIDS deaths ######################
    model_label = "AIDS deaths per 100000"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df, "{}{}_linreg_model.csv".format(results_dir, model_label))

    conn_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, conn_SSP, linreg_result)
    arkan_result = linreg_iowa_simulate_num_SSPs(model_feature, model_label, iowa_future_df, arkan_SSP, linreg_result)
    state_SSP = pd.concat([conn_result,arkan_result], axis=1)
    state_SSP.columns = [model_label, 'Arkansas simulation']
    state_SSP['Year'] = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    state_SSP.to_csv("linreg_state_simulation_{}.csv".format(model_label), index=False)
    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df)
    linreg_simulate_SSP_legality(model_feature, model_label, df)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df)
    

