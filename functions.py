import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import time
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics

import pyomo.environ as pyo
import pickle

from numpy.random import binomial, normal
from scipy.stats import bernoulli, binom

# Erzeuge Trainingsdatensatz mit Werbe- und Kaufhistorie
# Ziehe randomisiert für jedes Kaufverhalten 
# entsprechend Bestellungen als Trainingsdatensatz.
# Es werden gleichverteilt beide Fälle 
# also schicke Werbung oder auch nicht erzeugt.


def create_training_set(cust_behaviour, iterations=1000):
    orders = []

    for i in range(iterations):
        order = cust_behaviour.sample() 
        
        # Hälfte der Trainingsdaten bekommt Werbung
        if 0.5 > random.uniform(0,1):
            if order["p_prom"].iat[0] > random.uniform(0, 1):        
                order["prom"] = 1
                order["has_bought"] = 1
                order["size"] = calc_order_size(order["size_prom"])
            else:
                order["prom"] = 1
                order["has_bought"] = 0
                order["size"] = 0
        else:
            if order["p_no_prom"].iat[0] > random.uniform(0, 1):
                order["prom"] = 0
                order["has_bought"] = 1
                order["size"] = calc_order_size(order["size_no_prom"])
            else:
                order["prom"] = 0
                order["has_bought"] = 0
                order["size"] = 0
        if order["t1_buy"].iat[0] == 1:
            if order["t1_prom"].iat[0] == 1:
                order["t1_size"] = calc_order_size(order["size_prom"])
            else:
                order["t1_size"] = calc_order_size(order["size_no_prom"])
        else:
            order["t1_size"] = 0
        if order["t2_buy"].iat[0] == 1:
            if order["t2_prom"].iat[0] == 1:
                order["t2_size"] = calc_order_size(order["size_prom"])
            else:
                order["t2_size"] = calc_order_size(order["size_no_prom"])
        else:
            order["t2_size"] = 0
        if order["t3_buy"].iat[0] == 1:
            if order["t3_prom"].iat[0] == 1:
                order["t3_size"] = calc_order_size(order["size_prom"])
            else:
                order["t3_size"] = calc_order_size(order["size_no_prom"])
        else:
            order["t3_size"] = 0
        orders.append(order)
    return pd.concat(orders)


def load_models(label, path):
    model_clf_logit = pickle.load(open(path + 'logit_model_'+label+'.sav', 'rb'))
    model_clf_rf = pickle.load(open(path + 'random_forest_clf_'+label+'.sav', 'rb'))
    model_clf_gtb = pickle.load(open(path + 'gradient_boosting_clf_'+label+'.sav', 'rb'))

    model_regr_linear = pickle.load(open(path + 'linear_model_'+label+'.sav', 'rb'))
    model_regr_rf = pickle.load(open(path + 'random_forest_regr_'+label+'.sav', 'rb')) 
    model_regr_gtb = pickle.load(open(path + 'gradient_boosting_regr_'+label+'.sav', 'rb')) 

    return [model_clf_logit, model_clf_rf, model_clf_gtb, model_regr_linear, model_regr_rf, model_regr_gtb]


def build_models(orders, label, path, feature_set="full"):
    filenames = []

    if feature_set == "full":
        train_orders_features, feature_list_clf, feature_list = add_features(orders.copy())
    elif feature_set == "2_periods":
        train_orders_features, feature_list_clf, feature_list = add_features_only_2_periods(orders.copy())
    elif feature_set == "1_period":
        train_orders_features, feature_list_clf, feature_list = add_features_only_1_period(orders.copy())
    elif feature_set == "without_prom":
        train_orders_features, feature_list_clf, feature_list = add_features_without_prom_hist(orders.copy())

    y1 = train_orders_features["has_bought"].copy()
    X1 = train_orders_features[feature_list_clf].copy()

    df1 = train_orders_features[train_orders_features["has_bought"] == 1].copy()
    y2 = df1["size"]
    X2 = df1[feature_list]

    scaler = StandardScaler()
    lr = LogisticRegression()
    logit_model = Pipeline([('standardize', scaler), ('log_reg', lr)])
    logit_model.fit(X1, y1)
    filename = path + 'logit_model_'+label+'.sav'
    filenames.append(filename)
    pickle.dump(logit_model, open(filename, 'wb'))

    linear_model = LinearRegression() 
    linear_model.fit(X2, y2)
    filename = path + 'linear_model_'+label+'.sav'
    filenames.append(filename)
    pickle.dump(linear_model, open(filename, 'wb'))

    random_forest_clf = RandomForestClassifier()
    random_forest_clf.fit(X1, y1)
    filename = path + 'random_forest_clf_'+label+'.sav'
    filenames.append(filename)
    pickle.dump(random_forest_clf, open(filename, 'wb'))

    random_forest_regr = RandomForestRegressor()
    random_forest_regr.fit(X2, y2)
    filename = path + 'random_forest_regr_'+label+'.sav'
    filenames.append(filename)
    pickle.dump(random_forest_regr, open(filename, 'wb'))

    gradient_boosting_clf = GradientBoostingClassifier()
    gradient_boosting_clf.fit(X1, y1)
    filename = path + 'gradient_boosting_clf_'+label+'.sav'
    filenames.append(filename)
    pickle.dump(gradient_boosting_clf, open(filename, 'wb'))

    gradient_boosting_regr = GradientBoostingRegressor()
    gradient_boosting_regr.fit(X2, y2)
    filename = path + 'gradient_boosting_regr_'+label+'.sav'
    filenames.append(filename)
    pickle.dump(gradient_boosting_regr, open(filename, 'wb'))

    return filenames


def get_periods_since_last_actions(t1=0, t2=0, t3=0):
    if t3 == 1:
        return 1
    if t2 == 1:
        return 2
    if t1 == 1:
        return 3

    return 5

def get_size_mean(size1=0, size2=0, size3=0):
    sum = 0
    count = 0
    if size1 > 0:
        count += 1
        sum += size1
    if size2 > 0:
        count += 1
        sum += size2
    if size3 > 0:
        count += 1
        sum += size3

    if count > 0:
        return sum/count
    else:
        return 0


def add_features(df):
    # TODO: add choosen prom to prom_xxx history --> prom_xxxx    
    df["periods_since_last_buy"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_buy"], row["t2_buy"], row["t3_buy"]), axis=1)
    df["periods_since_last_prom"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_prom"], row["t2_prom"], row["t3_prom"]), axis=1)
    df["size_mean"] = df.apply(lambda row : get_size_mean(row["t1_size"], row["t2_size"], row["t3_size"]), axis=1)
    df["prom_1111"] = (df["prom"] == 1) & (df["t3_prom"] == 1) & (df["t2_prom"] == 1) & (df["t1_prom"] == 1) 
    df["prom_1011"] = (df["prom"] == 1) & (df["t3_prom"] == 0) & (df["t2_prom"] == 1) & (df["t1_prom"] == 1) 
    df["prom_1101"] = (df["prom"] == 1) & (df["t3_prom"] == 1) & (df["t2_prom"] == 0) & (df["t1_prom"] == 1) 
    df["prom_1110"] = (df["prom"] == 1) & (df["t3_prom"] == 1) & (df["t2_prom"] == 1) & (df["t1_prom"] == 0) 
    df["prom_1001"] = (df["prom"] == 1) & (df["t3_prom"] == 0) & (df["t2_prom"] == 0) & (df["t1_prom"] == 1) 
    df["prom_1010"] = (df["prom"] == 1) & (df["t3_prom"] == 0) & (df["t2_prom"] == 1) & (df["t1_prom"] == 0) 
    df["prom_1100"] = (df["prom"] == 1) & (df["t3_prom"] == 1) & (df["t2_prom"] == 0) & (df["t1_prom"] == 0) 
    df["prom_1000"] = (df["prom"] == 1) & (df["t3_prom"] == 0) & (df["t2_prom"] == 0) & (df["t1_prom"] == 0) 

    df["prom_0111"] = (df["prom"] == 0) & (df["t3_prom"] == 1) & (df["t2_prom"] == 1) & (df["t1_prom"] == 1) 
    df["prom_0011"] = (df["prom"] == 0) & (df["t3_prom"] == 0) & (df["t2_prom"] == 1) & (df["t1_prom"] == 1) 
    df["prom_0101"] = (df["prom"] == 0) & (df["t3_prom"] == 1) & (df["t2_prom"] == 0) & (df["t1_prom"] == 1) 
    df["prom_0110"] = (df["prom"] == 0) & (df["t3_prom"] == 1) & (df["t2_prom"] == 1) & (df["t1_prom"] == 0) 
    df["prom_0001"] = (df["prom"] == 0) & (df["t3_prom"] == 0) & (df["t2_prom"] == 0) & (df["t1_prom"] == 1) 
    df["prom_0010"] = (df["prom"] == 0) & (df["t3_prom"] == 0) & (df["t2_prom"] == 1) & (df["t1_prom"] == 0) 
    df["prom_0100"] = (df["prom"] == 0) & (df["t3_prom"] == 1) & (df["t2_prom"] == 0) & (df["t1_prom"] == 0) 
    df["prom_0000"] = (df["prom"] == 0) & (df["t3_prom"] == 0) & (df["t2_prom"] == 0) & (df["t1_prom"] == 0) 

    feature_list = ["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom", "t1_size", "t2_size", "t3_size", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_1111", "prom_1011", "prom_1101", "prom_1110", "prom_1001", "prom_1010", "prom_1100", "prom_1000",
        "prom_0111", "prom_0011", "prom_0101", "prom_0110", "prom_0001", "prom_0010", "prom_0100", "prom_0000",
        "prom", "size_mean"]
    feature_list_clf = ["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_1111", "prom_1011", "prom_1101", "prom_1110", "prom_1001", "prom_1010", "prom_1100", "prom_1000",
        "prom_0111", "prom_0011", "prom_0101", "prom_0110", "prom_0001", "prom_0010", "prom_0100", "prom_0000",
        "prom"]
    
    return df, feature_list_clf, feature_list


def add_features_without_prom_hist(df):
    # TODO: add choosen prom to prom_xxx history --> prom_xxxx    
    df["periods_since_last_buy"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_buy"], row["t2_buy"], row["t3_buy"]), axis=1)
    df["periods_since_last_prom"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_prom"], row["t2_prom"], row["t3_prom"]), axis=1)
    df["size_mean"] = df.apply(lambda row : get_size_mean(row["t1_size"], row["t2_size"], row["t3_size"]), axis=1)

    feature_list = ["t1_buy", "t2_buy", "t3_buy", "t1_size", "t2_size", "t3_size", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom", "size_mean"]
    feature_list_clf = ["t1_buy", "t2_buy", "t3_buy", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom"]
    
    return df, feature_list_clf, feature_list


def add_features_only_2_periods(df):
    # TODO: add choosen prom to prom_xxx history --> prom_xxxx    
    df["periods_since_last_buy"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_buy"], row["t2_buy"]), axis=1)
    df["periods_since_last_prom"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_prom"], row["t2_prom"]), axis=1)
    df["size_mean"] = df.apply(lambda row : get_size_mean(row["t1_size"], row["t2_size"]), axis=1)

    df["prom_111"] = (df["prom"] == 1) & (df["t1_prom"] == 1) & (df["t2_prom"] == 1) 
    df["prom_101"] = (df["prom"] == 1) & (df["t1_prom"] == 0) & (df["t2_prom"] == 1) 
    df["prom_100"] = (df["prom"] == 1) & (df["t1_prom"] == 0) & (df["t2_prom"] == 0) 
    df["prom_110"] = (df["prom"] == 1) & (df["t1_prom"] == 1) & (df["t2_prom"] == 0) 
    df["prom_011"] = (df["prom"] == 0) & (df["t1_prom"] == 1) & (df["t2_prom"] == 1) 
    df["prom_001"] = (df["prom"] == 0) & (df["t1_prom"] == 0) & (df["t2_prom"] == 1) 
    df["prom_010"] = (df["prom"] == 0) & (df["t1_prom"] == 1) & (df["t2_prom"] == 0) 
    df["prom_000"] = (df["prom"] == 0) & (df["t1_prom"] == 0) & (df["t2_prom"] == 0) 

    feature_list = ["t1_buy", "t2_buy", "t1_prom", "t2_prom", "t1_size", "t2_size", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_111", "prom_101", "prom_110", "prom_100",
        "prom_011", "prom_001", "prom_010", "prom_000",
        "prom", "size_mean"]
    feature_list_clf = ["t1_buy", "t2_buy", "t1_prom", "t2_prom", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_111", "prom_101", "prom_110", "prom_100",
        "prom_011", "prom_001", "prom_010", "prom_000",
        "prom"]
    
    return df, feature_list_clf, feature_list


def add_features_only_1_period(df):
    # TODO: add choosen prom to prom_xxx history --> prom_xxxx    
    df["periods_since_last_buy"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_buy"]), axis=1)
    df["periods_since_last_prom"] = df.apply(lambda row : get_periods_since_last_actions(row["t1_prom"]), axis=1)
    df["size_mean"] = df.apply(lambda row : get_size_mean(row["t1_size"]), axis=1)

    df["prom_11"] = (df["prom"] == 1) & (df["t1_prom"] == 1)  
    df["prom_10"] = (df["prom"] == 1) & (df["t1_prom"] == 0) 
    df["prom_01"] = (df["prom"] == 0) & (df["t1_prom"] == 1) 
    df["prom_00"] = (df["prom"] == 0) & (df["t1_prom"] == 0) 

    feature_list = ["t1_buy", "t1_prom", "t1_size", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_11", "prom_10", "prom_01", "prom_00",
        "prom", "size_mean"]
    feature_list_clf = ["t1_buy", "t1_prom", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_11", "prom_10", "prom_01", "prom_00",
        "prom"]
    
    return df, feature_list_clf, feature_list

def create_cust_behaviour_new():
    # minimal: unterschiedliche Beispiele die ähnlich sind
    
    # bewusstheit reinbringen
    # was passiert bei größerem Werbenutzen und kleinerer KWahrscheinlichkeit ohne Werbung
    possibilities = [[0,1], [0,1], [0,1]]
    columns = ["t1_buy", "t2_buy", "t3_buy"]
    cust_buy_behaviour3 = pd.DataFrame(product(*possibilities), columns = columns)
    cust_buy_behaviour3["p_no_prom"] = np.random.uniform(0.05, 0.3, 8)
    cust_buy_behaviour3["size_no_prom"] = np.random.uniform(15, 150, 8)

    possibilities = [[0,1], [0,1], [0,1]]
    columns = ["t1_prom", "t2_prom", "t3_prom"]
    cust_prom_behaviour3 = pd.DataFrame(product(*possibilities), columns = columns)
    cust_prom_behaviour3["p_no_prom"] =  pd.Series([0.1, 0.05, 0.4, 0.3, 0.5, 0.35, 0.7, 0.8])
    cust_prom_behaviour3["size_no_prom"] = np.random.uniform(15, 150, 8)

    cust_behaviour3 = pd.merge(cust_buy_behaviour3, cust_prom_behaviour3, how="cross")
    cust_behaviour3["p_no_prom"] = (cust_behaviour3["p_no_prom_x"] + cust_behaviour3["p_no_prom_y"]) / 2
    cust_behaviour3["size_no_prom"] = (cust_behaviour3["size_no_prom_x"] + cust_behaviour3["size_no_prom_y"]) / 2
    cust_behaviour3["p_prom"] = cust_behaviour3["p_no_prom"] * np.random.uniform(0.3, 3.0, 64)
    cust_behaviour3["size_prom"] = cust_behaviour3["size_no_prom"] * np.random.uniform(0.4, 3.0, 64)

    cust_behaviour3 = cust_behaviour3.drop(['p_no_prom_y', 'p_no_prom_x', 'size_no_prom_x', 'size_no_prom_y'], axis=1)

    return cust_behaviour3

def create_cust_behaviour_3():
    # minimal: unterschiedliche Beispiele die ähnlich sind
    
    # bewusstheit reinbringen
    # was passiert bei größerem Werbenutzen und kleinerer KWahrscheinlichkeit ohne Werbung
    possibilities = [[0,1], [0,1], [0,1]]
    columns = ["t1_buy", "t2_buy", "t3_buy"]
    cust_buy_behaviour3 = pd.DataFrame(product(*possibilities), columns = columns)
    cust_buy_behaviour3["p_no_prom"] = pd.Series([0.1, 0.05, 0.4, 0.3, 0.5, 0.35, 0.7, 0.8])
    cust_buy_behaviour3["size_no_prom"] = np.random.uniform(15, 150, 8)

    possibilities = [[0,1], [0,1], [0,1]]
    columns = ["t1_prom", "t2_prom", "t3_prom"]
    cust_prom_behaviour3 = pd.DataFrame(product(*possibilities), columns = columns)
    cust_prom_behaviour3["p_no_prom"] =  pd.Series([0.1, 0.05, 0.4, 0.3, 0.5, 0.35, 0.7, 0.8])
    cust_prom_behaviour3["size_no_prom"] = np.random.uniform(15, 150, 8)

    cust_behaviour3 = pd.merge(cust_buy_behaviour3, cust_prom_behaviour3, how="cross")
    cust_behaviour3["p_no_prom"] = (cust_behaviour3["p_no_prom_x"] + cust_behaviour3["p_no_prom_y"]) / 2
    cust_behaviour3["size_no_prom"] = (cust_behaviour3["size_no_prom_x"] + cust_behaviour3["size_no_prom_y"]) / 2
    cust_behaviour3["p_prom"] = cust_behaviour3["p_no_prom"] * np.random.uniform(0.9, 1.2, 64)
    cust_behaviour3["size_prom"] = cust_behaviour3["size_no_prom"] * np.random.uniform(0.9, 1.3, 64)

    cust_behaviour3 = cust_behaviour3.drop(['p_no_prom_y', 'p_no_prom_x', 'size_no_prom_x', 'size_no_prom_y'], axis=1)

    return cust_behaviour3

### EXPORT CUST BEHAVIOUR WITH EARNINGS and not with PROFIT
#### ampl input format w1,w2,w3,a1,a2,a3,w,  P-Wert,  R-Wert 
###def export_cust_behaviour_for_ampl(cust_behaviour, fullfilepath):
###    cust_behaviour_prom = cust_behaviour[['t1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 'p_prom', 'size_prom']]
###    cust_behaviour_prom = cust_behaviour_prom.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_prom": "P", "size_prom": "R"})
###    cust_behaviour_prom["w"] = 1
###    cust_behaviour_no_prom = cust_behaviour[['t1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 'p_no_prom', 'size_no_prom']]
###    cust_behaviour_no_prom = cust_behaviour_no_prom.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_no_prom": "P", "size_no_prom": "R"})
###    cust_behaviour_no_prom["w"] = 0
###    cust_behaviour_ampl = pd.concat([cust_behaviour_prom, cust_behaviour_no_prom]) 
###    cust_behaviour_ampl = cust_behaviour_ampl[["w1", "w2", "w3", "a1" ,"a2" ,"a3", "w", "P", "R"]]
###    cust_behaviour_ampl.to_csv(fullfilepath, index=False)
###    print(cust_behaviour_ampl)

def predict_test_set(label, path, test_set, feature_list, cust_behaviour):
    model_clf_logit, model_clf_rf, model_clf_gtb, model_regr_linear, model_regr_rf, model_regr_gtb = load_models(label, path)
    
    feature_list_clf = ["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom", 
        "periods_since_last_buy", "periods_since_last_prom", 
        "prom_1111", "prom_1011", "prom_1101", "prom_1110", "prom_1001", "prom_1010", "prom_1100", "prom_1000",
        "prom_0111", "prom_0011", "prom_0101", "prom_0110", "prom_0001", "prom_0010", "prom_0100", "prom_0000",
        "prom"]
    y_clf = test_set["has_bought"]
    X_clf = test_set[feature_list_clf]

    # for regression use only the size of orders (exclude customers without buy)
    df_buys = test_set[test_set["has_bought"] == 1]
    y_regr = df_buys["size"]
    X_regr = df_buys[feature_list]

    # compare predictions between models
    eval_list = ["prom", "has_bought", "logit_pred", "rf_pred", "gtb_pred", "true_prob", "null_model", "logit_prob", "rf_prob", "gtb_prob", "size", "linear_size", "rf_size", "gtb_size"] # "true_prob_no_prom", "true_prob_with_prom"

    #df = test_set.merge(cust_behaviour, how='left', on=["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom"])["p_no_prom_x"]
    test_set["true_prob_no_prom"] = test_set.merge(cust_behaviour, how='left', on=["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom"])["p_no_prom_x"]
    test_set["true_prob_with_prom"] = test_set.merge(cust_behaviour, how='left', on=["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom"])["p_prom_x"]
    test_set["true_prob"] = test_set["true_prob_no_prom"] 

    test_set.loc[test_set["prom"] == 1, "true_prob"] = test_set[test_set["prom"] == 1]["true_prob_with_prom"]

    test_set["logit_pred"] = model_clf_logit.predict(X_clf)
    test_set["rf_pred"] = model_clf_rf.predict(X_clf)
    test_set["gtb_pred"] = model_clf_gtb.predict(X_clf)

    test_set["null_model"] = np.mean(test_set["has_bought"])

    test_set["logit_prob"] = pd.DataFrame(model_clf_logit.predict_proba(X_clf))[1].to_numpy()
    test_set["rf_prob"] = pd.DataFrame(model_clf_rf.predict_proba(X_clf))[1].to_numpy() 
    test_set["gtb_prob"] = pd.DataFrame(model_clf_gtb.predict_proba(X_clf))[1].to_numpy()  

    test_set["linear_size"] = model_regr_linear.predict(X_clf)
    test_set["rf_size"] = model_regr_rf.predict(X_clf)
    test_set["gtb_size"] = model_regr_gtb.predict(X_clf)
    return test_set


def evaluate_models(label, path, test_orders_features, feature_list, feature_list_clf, training_set_size, cust_behaviour, models):
    model_clf_logit, model_clf_rf, model_clf_gtb, model_regr_linear, model_regr_rf, model_regr_gtb = load_models(label, path)
    orders = test_orders_features.copy()

    y_clf = orders["has_bought"]
    X_clf = orders[feature_list_clf]

    # for regression use only the size of orders (exclude customers without buy)
    df_buys = orders[orders["has_bought"] == 1]
    y_regr = df_buys["size"]
    X_regr = df_buys[feature_list]

    orders["logit_pred"] = model_clf_logit.predict(X_clf)
    orders["rf_pred"] = model_clf_rf.predict(X_clf)
    orders["gtb_pred"] = model_clf_gtb.predict(X_clf)

    orders["null_model"] = np.mean(orders["has_bought"])

    orders["logit_prob"] = pd.DataFrame(model_clf_logit.predict_proba(X_clf))[1].to_numpy()
    orders["rf_prob"] = pd.DataFrame(model_clf_rf.predict_proba(X_clf))[1].to_numpy() 
    orders["gtb_prob"] = pd.DataFrame(model_clf_gtb.predict_proba(X_clf))[1].to_numpy() 
    
    orders["linear_size"] = model_regr_linear.predict(orders[feature_list])
    orders["rf_size"] = model_regr_rf.predict(orders[feature_list])
    orders["gtb_size"] = model_regr_gtb.predict(orders[feature_list])

    
    orders = orders.merge(cust_behaviour, how='left', on=["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom"])
    orders["true_prob"] = orders["p_no_prom_y"]
    orders.loc[orders["prom"] == 1, "true_prob"] = orders[orders["prom"] == 1]["p_prom_y"]

    eval_metrics = ["accuracy", "precision", "recall", "f1_score"]

    accuracy = orders[orders["has_bought"] == orders["logit_pred"]]["logit_pred"].count() / orders["logit_pred"].count()
    precision = orders.loc[(orders["has_bought"] == 1) & (orders["logit_pred"] == 1), ["logit_pred"]]["logit_pred"].count() / orders[orders["has_bought"] == 1]["logit_pred"].count()
    recall = orders.loc[(orders["has_bought"] == 1) & (orders["logit_pred"] == 1), ["logit_pred"]]["logit_pred"].count() / orders[orders["logit_pred"] == 1]["logit_pred"].count()
    f1_score = 2 * (precision * recall) / (precision + recall) 
    logit_eval = [accuracy, precision, recall, f1_score]

    # "rf_pred"
    accuracy = orders[orders["has_bought"] == orders["rf_pred"]]["rf_pred"].count() / orders["rf_pred"].count()
    precision = orders.loc[(orders["has_bought"] == 1) & (orders["rf_pred"] == 1), ["rf_pred"]]["rf_pred"].count() / orders[orders["has_bought"] == 1]["rf_pred"].count()
    recall = orders.loc[(orders["has_bought"] == 1) & (orders["rf_pred"] == 1), ["rf_pred"]]["rf_pred"].count() / orders[orders["rf_pred"] == 1]["rf_pred"].count()
    f1_score = 2 * (precision * recall) / (precision + recall) 
    rf_eval = [accuracy, precision, recall, f1_score]

    # "gtb_pred"
    accuracy = orders[orders["has_bought"] == orders["gtb_pred"]]["gtb_pred"].count() / orders["gtb_pred"].count()
    precision = orders.loc[(orders["has_bought"] == 1) & (orders["gtb_pred"] == 1), ["gtb_pred"]]["gtb_pred"].count() / orders[orders["has_bought"] == 1]["gtb_pred"].count()
    recall = orders.loc[(orders["has_bought"] == 1) & (orders["gtb_pred"] == 1), ["gtb_pred"]]["gtb_pred"].count() / orders[orders["gtb_pred"] == 1]["gtb_pred"].count()
    f1_score = 2 * (precision * recall) / (precision + recall) 
    gtb_eval = [accuracy, precision, recall, f1_score]

    accuracy_metrics = pd.DataFrame([logit_eval, rf_eval, gtb_eval], index=["logit", "rf_clf", "gtb_clf"], columns=eval_metrics)
    accuracy_metrics = accuracy_metrics.transpose().round(5)
    accuracy_metrics["kpi"] = accuracy_metrics.index
    accuracy_metrics["training_set_size"] = training_set_size
    models = pd.concat([models, accuracy_metrics])
    
    eval_metrics = ["Explained Variance", "Mean Squared Error", "Root Mean Squared Error", "Mean Absolut Error", "R2"]

    var = metrics.explained_variance_score(orders["true_prob"], orders["logit_prob"])
    mse = metrics.mean_squared_error(orders["true_prob"], orders["logit_prob"])
    rmse = metrics.mean_squared_error(orders["true_prob"], orders["logit_prob"], squared=False)
    msa = metrics.mean_absolute_error(orders["true_prob"], orders["logit_prob"])
    r2 = metrics.r2_score(orders["true_prob"], orders["logit_prob"])
    logit_eval = [var, mse, rmse, msa, r2]

    var = metrics.explained_variance_score(orders["true_prob"], orders["rf_prob"])
    mse = metrics.mean_squared_error(orders["true_prob"], orders["rf_prob"])
    rmse = metrics.mean_squared_error(orders["true_prob"], orders["rf_prob"], squared=False)
    msa = metrics.mean_absolute_error(orders["true_prob"], orders["rf_prob"])
    r2 = metrics.r2_score(orders["true_prob"], orders["rf_prob"])
    rf_eval = [var, mse, rmse, msa, r2]

    var = metrics.explained_variance_score(orders["true_prob"], orders["gtb_prob"])
    mse = metrics.mean_squared_error(orders["true_prob"], orders["gtb_prob"])
    rmse = metrics.mean_squared_error(orders["true_prob"], orders["gtb_prob"], squared=False)
    msa = metrics.mean_absolute_error(orders["true_prob"], orders["gtb_prob"])
    r2 = metrics.r2_score(orders["true_prob"], orders["gtb_prob"])
    gtb_eval = [var, mse, rmse, msa, r2]

    m = pd.DataFrame([logit_eval, rf_eval, gtb_eval], index=["logit", "rf_clf", "gtb_clf"], columns=eval_metrics)
    m = m.transpose().round(5)
    m["kpi"] = m.index
    m["training_set_size"] = training_set_size
    models = pd.concat([models, m])

    eval_metrics = ["Explained Variance", "Mean Squared Error", "Root Mean Squared Error", "Mean Absolut Error", "R2"]

    var = metrics.explained_variance_score(y_regr, model_regr_linear.predict(X_regr))
    mse = metrics.mean_squared_error(y_regr, model_regr_linear.predict(X_regr))
    rmse = metrics.mean_squared_error(y_regr, model_regr_linear.predict(X_regr), squared=False)
    msa = metrics.mean_absolute_error(y_regr, model_regr_linear.predict(X_regr))
    r2 = metrics.r2_score(y_regr, model_regr_linear.predict(X_regr))
    linear_eval = [var, mse, rmse, msa, r2]

    var = metrics.explained_variance_score(y_regr, model_regr_rf.predict(X_regr))
    mse = metrics.mean_squared_error(y_regr, model_regr_rf.predict(X_regr))
    rmse = metrics.mean_squared_error(y_regr, model_regr_rf.predict(X_regr), squared=False)
    msa = metrics.mean_absolute_error(y_regr, model_regr_rf.predict(X_regr))
    r2 = metrics.r2_score(y_regr, model_regr_rf.predict(X_regr))
    rf_eval = [var, mse, rmse, msa, r2]

    var = metrics.explained_variance_score(y_regr, model_regr_gtb.predict(X_regr))
    mse = metrics.mean_squared_error(y_regr, model_regr_gtb.predict(X_regr))
    rmse = metrics.mean_squared_error(y_regr, model_regr_gtb.predict(X_regr), squared=False)
    msa = metrics.mean_absolute_error(y_regr, model_regr_gtb.predict(X_regr))
    r2 = metrics.r2_score(y_regr, model_regr_gtb.predict(X_regr))
    gtb_eval = [var, mse, rmse, msa, r2]
    
    m = pd.DataFrame([linear_eval, rf_eval, gtb_eval], index=["linear", "rf_regr", "gtb_regr"], columns=eval_metrics)
    m = m.transpose().round(5)
    m["kpi"] = m.index
    m["training_set_size"] = training_set_size
    models = pd.concat([models, m])

    return models, orders

    


# https://machinelearningknowledge.ai/python-sklearn-logistic-regression-tutorial-with-example/
def build_logit_model(y, X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # TODO: mit intercept oder ohne?
    scaler = StandardScaler()
    lr = LogisticRegression()
    model1 = Pipeline([('standardize', scaler),
                        ('log_reg', lr)])
    model1.fit(X_train, y_train)

    y_train_hat = model1.predict(X_train)
    y_train_hat_probs = model1.predict_proba(X_train)[:,1]

    train_accuracy = accuracy_score(y_train, y_train_hat)*100
    train_auc_roc = roc_auc_score(y_train, y_train_hat_probs)*100

    print('Confusion matrix:\n', confusion_matrix(y_train, y_train_hat))
    print('Training AUC: %.4f %%' % train_auc_roc)
    print('Training accuracy: %.4f %%' % train_accuracy)

    y_test_hat = model1.predict(X_test)
    y_test_hat_probs = model1.predict_proba(X_test)[:,1]

    test_accuracy = accuracy_score(y_test, y_test_hat)*100
    test_auc_roc = roc_auc_score(y_test, y_test_hat_probs)*100

    print('Confusion matrix:\n', confusion_matrix(y_test, y_test_hat))
    print('Testing AUC: %.4f %%' % test_auc_roc)
    print('Testing accuracy: %.4f %%' % test_accuracy) 

    print(classification_report(y_test, y_test_hat, digits=6))

    return model1


# https://machinelearningknowledge.ai/linear-regression-in-python-sklearn-with-example/
def build_linear_model(y, X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    regr = LinearRegression() 
    regr.fit(X_train, y_train)

    train_score = regr.score(X_train, y_train)
    print("R2 - training score: ", train_score)

    test_score = regr.score(X_test, y_test)
    print("R2 - test score:", test_score)

    return regr


def create_cust_with_initial_behaviour(cust_count, first_cust_nr):
    cust_nrs = list(range(first_cust_nr, first_cust_nr + cust_count))
    t1_buy = random.choices([0, 1], weights=[50, 50], k=cust_count)
    t2_buy = random.choices([0, 1], weights=[50, 50], k=cust_count)
    t3_buy = random.choices([0, 1], weights=[50, 50], k=cust_count)
    t1_prom = random.choices([0, 1], weights=[50, 50], k=cust_count)
    t2_prom = random.choices([0, 1], weights=[50, 50], k=cust_count)
    t3_prom = random.choices([0, 1], weights=[50, 50], k=cust_count)
    t1_size = np.random.uniform(15, 150, cust_count)
    t2_size = np.random.uniform(15, 150, cust_count)
    t3_size = np.random.uniform(15, 150, cust_count)

    dict = {
        "cust_nrs": pd.Series(cust_nrs, index=cust_nrs),
        "t1_buy": pd.Series(t1_buy, index=cust_nrs),
        "t2_buy": pd.Series(t2_buy, index=cust_nrs),
        "t3_buy": pd.Series(t3_buy, index=cust_nrs),
        "t1_prom": pd.Series(t1_prom, index=cust_nrs),
        "t2_prom": pd.Series(t2_prom, index=cust_nrs),
        "t3_prom": pd.Series(t3_prom, index=cust_nrs),
        "t1_size": pd.Series(t1_size, index=cust_nrs),
        "t2_size": pd.Series(t2_size, index=cust_nrs),
        "t3_size": pd.Series(t3_size, index=cust_nrs)
    }

    customers = pd.DataFrame(dict, columns=['cust_nrs', 't1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 't1_size', 't2_size', 't3_size'])
    customers.loc[customers["t1_buy"] == 0, ["t1_size"]] = 0
    customers.loc[customers["t2_buy"] == 0, ["t2_size"]] = 0
    customers.loc[customers["t3_buy"] == 0, ["t3_size"]] = 0

    return customers


# Optimale Strategie ohne Budgetrestriktionen und Kosten (berechnet mit den tatsächlichen Wahrscheinlichkeiten)
# "Nehme den höchsten erwarteten Umsatz"
def calc_true_expected_profit_for_opt_strategy(customers, fix_order_costs, margin, cost_per_prom):
    customers["rev_expected_prom"] = customers["size_prom"] * customers["p_prom"]
    customers['profit_expected_prom'] = customers["size_prom"] * customers["p_prom"] * margin - customers["p_prom"] * fix_order_costs - cost_per_prom

    customers["rev_expected_no_prom"] = customers["size_no_prom"] * customers["p_no_prom"]
    customers['profit_expected_no_prom'] = customers["size_no_prom"] * customers["p_no_prom"] * margin - customers["p_no_prom"] * fix_order_costs

    return customers


def get_optimal_profit(df, model_training_set_size, min_proms, max_proms, optimal_label):
    e3 = df[df["min_proms"] == min_proms]
    e3 = e3[e3["max_proms"] == max_proms]
    e3 = e3[e3["models"] == optimal_label]
    if model_training_set_size != "unset":
        e3 = e3[e3["model_training_set_size"] == model_training_set_size]
    return float(e3["profit"])

    
# Generate Orders
# Realisiere für jeden Kunden entsprechend der optimalen Strategie die Bestellungen.
# Dabei wird die Kaufhistorie für das Kaufverhalten entsprechend genutzt.
# return df enthält die Kauf- und Werbehistorie für die nächste Werbeperiode
def generate_orders(customers, margin, fix_order_costs, cost_per_prom):
    orders = []

    for ind in customers.index:
        order = {
            "cust_nrs" : customers['cust_nrs'][ind],
            "send_prom" : customers['send_prom'][ind],
            "t1_buy" : customers['t2_buy'][ind],
            "t2_buy" : customers['t3_buy'][ind],
            "t1_prom" : customers['t2_prom'][ind],
            "t2_prom" : customers['t3_prom'][ind],
            "t1_size" : customers['t2_size'][ind],
            "t2_size" : customers['t3_size'][ind],
        }
        if customers['send_prom'][ind]:
            if customers['p_prom'][ind] > random.uniform(0, 1):
                order["has_prom"] = 1
                order["has_bought"] = 1
                order["size"] = calc_order_size(float(customers["size_prom"][ind]))
                order["t3_prom"] = 1
                order["t3_buy"] = 1
                order["t3_size"] = order["size"]
            else:
                order["has_prom"] = 1
                order["has_bought"] = 0
                order["size"] = 0
                order["t3_prom"] = 1
                order["t3_buy"] = 0
                order["t3_size"] = 0
        else:
            if customers['p_no_prom'][ind] > random.uniform(0, 1):
                order["has_prom"] = 0
                order["has_bought"] = 1
                order["size"] = calc_order_size(float(customers["size_no_prom"][ind]))    
                order["t3_prom"] = 0
                order["t3_buy"] = 1
                order["t3_size"] = order["size"]
            else:
                order["has_prom"] = 0
                order["has_bought"] = 0
                order["size"] = 0
                order["t3_prom"] = 0
                order["t3_buy"] = 0
                order["t3_size"] = 0
        
        order["size_prom"] = customers["size_prom"][ind]
        order["p_prom"] = customers['p_prom'][ind]
        order["profit_expected_prom"] = customers["size_prom"][ind] * customers["p_prom"][ind] * margin - customers["p_prom"][ind] * fix_order_costs - cost_per_prom
        order["size_no_prom"] = customers["size_no_prom"][ind]
        order["p_no_prom"] = customers['p_no_prom'][ind]
        order["profit_expected_no_prom"] = customers["size_no_prom"][ind] * customers["p_no_prom"][ind] * margin - customers["p_no_prom"][ind] * fix_order_costs
        if customers['send_prom'][ind]:
            order["exp_profit"] = order["profit_expected_prom"]
        else:
            order["exp_profit"] = order["profit_expected_no_prom"]
        orders.append(order)

    return pd.DataFrame(orders)


def calc_order_size(avg_order_size):
    avg_order_size = float(avg_order_size)
    if avg_order_size < 35:
        return np.random.uniform(5, (avg_order_size * 2) - 5)
    else:
        return np.random.uniform(avg_order_size - 15, avg_order_size + 15)


def predict_cust_behaviour_with_clf_and_regr_model(model_clf, model_regr, feature_set, cust_behaviour, customers, fix_order_costs, margin, cost_per_prom):
    customers["prom"] = 0
    if feature_set == "full":
        customers_regr, feature_list_clf, feature_list = add_features(customers)
    elif feature_set == "2_periods":
        customers_regr, feature_list_clf, feature_list = add_features_only_2_periods(customers)
    elif feature_set == "1_period":
        customers_regr, feature_list_clf, feature_list = add_features_only_1_period(customers)
    elif feature_set == "without_prom":
        customers_regr, feature_list_clf, feature_list = add_features_without_prom_hist(customers)

    X0_clf = customers_regr[feature_list_clf].copy()
    X0_regr = customers_regr[feature_list].copy()
    customers_regr["p_no_prom_expected"] = pd.DataFrame(model_clf.predict_proba(X0_clf))[1].to_numpy()
    customers_regr["size_no_prom_expected"] = model_regr.predict(X0_regr) 
    customers_regr["rev_expected_no_prom"] = customers_regr["size_no_prom_expected"] * customers_regr["p_no_prom_expected"]
    customers_regr['profit_expected_no_prom'] = customers_regr['rev_expected_no_prom'] * margin - fix_order_costs


    customers_regr["prom"] = 1
    X1_clf = customers_regr[feature_list_clf].copy()
    X1_regr = customers_regr[feature_list].copy()
    customers_regr["p_prom_expected"] = pd.DataFrame(model_clf.predict_proba(X1_clf))[1].to_numpy()
    customers_regr["size_prom_expected"] = model_regr.predict(X1_regr) 
    customers_regr["rev_expected_prom"] = customers_regr["size_prom_expected"] * customers_regr["p_prom_expected"]
    customers_regr['profit_expected_prom'] = customers_regr['rev_expected_prom'] * margin - fix_order_costs - cost_per_prom
    
    return customers_regr


def decide_prom(customer):
    customer["send_prom"] = False
    customer["send_prom"] = customer["profit_expected_prom"] > customer["profit_expected_no_prom"]

    return customer

def send_prom_to_hprob(c, min_proms, max_proms):
    c = c.sort_values("p_prom", ascending=False, ignore_index=True)
    c["send_prom"] = 0
    c.loc[0:(max_proms-1),"send_prom"] = 1
    return c

def send_prom_to_hs(c, min_proms, max_proms):
    c = c.sort_values("size_prom", ascending=False, ignore_index=True)
    c["send_prom"] = 0
    c.loc[0:(max_proms-1),"send_prom"] = 1
    return c

# höchste lowest q value kipppunkt
def send_prom_to_highest_kipppunkt(c, min_proms, max_proms):
    c = c.sort_values("kipppunkt", ascending=False, ignore_index=True)
    c["send_prom"] = 0
    c.loc[0:(max_proms-1),"send_prom"] = 1
    return c

# höchste Q-Value-Differenz
def send_prom_to_highest_q_value_diff(cust, min_proms, max_proms, c=1):
    cust = cust.sort_values("q_value_diff_c"+str(c), ascending=False, ignore_index=True)
    cust["send_prom"] = 0
    cust.loc[0:(max_proms-1),"send_prom"] = 1
    return cust

# höchster erwarteter Gewinn mit Werbung - highest profit
def send_prom_to_hprofit(c, min_proms, max_proms):
    c = c.sort_values("profit_expected_prom", ascending=False, ignore_index=True)
    c["send_prom"] = 0
    c.loc[0:(max_proms-1),"send_prom"] = 1
    return c

def send_prom_to_lb(c, min_proms, max_proms):
    c = c.sort_values(["t3_buy", "t2_buy", "t2_buy"], ascending=False, ignore_index=True)
    c["send_prom"] = 0
    c.loc[0:(max_proms-1),"send_prom"] = 1
    return c

def send_prom_to_nb(c, min_proms, max_proms):
    c = c.sort_values(["t3_buy", "t2_buy", "t2_buy"], ascending=True, ignore_index=True)
    c["send_prom"] = 0
    c.loc[0:(max_proms-1),"send_prom"] = 1
    return c

def send_prom_to_random(c, min_proms, max_proms):
    no_proms = c.shape[0] - max_proms
    prom = np.array([1]*max_proms + [0]*no_proms)
    np.random.shuffle(prom)
    c["send_prom"] = prom.tolist()
    return c

def decide_prom_with_lp(customer, max_proms, min_proms):
    C = list(customer.index)
    exp_prom = {(c) : customer.at[c, 'profit_expected_prom'] for c in C}
    exp_no_prom = {(c) : customer.at[c, 'profit_expected_no_prom'] for c in C}    

    model = lp_create_promotions_model(C, exp_prom, exp_no_prom, max_proms, min_proms)

    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    customer["send_prom"] = model.x.get_values().values()
    customer["send_prom"] = customer["send_prom"].astype('bool')

    return customer

def lp_create_promotions_model(C, exp_prom, exp_no_prom, max_proms, min_proms):
    model = pyo.ConcreteModel(name="(Customer Selection Policy)")
    model.x = pyo.Var(C, within=pyo.Binary)

    def obj_rule(mdl):
        return sum(mdl.x[c]*exp_prom[c] - (mdl.x[c]-1)*exp_no_prom[c] for c in C)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    def num_promotions_rule(mdl):
        return sum(mdl.x[c] for c in C) <= max_proms
    model.num_promotions = pyo.Constraint(rule=num_promotions_rule)

    def num_promotions_rule_min(mdl):
        return sum(mdl.x[c] for c in C) >= min_proms
    model.num_promotions_min = pyo.Constraint(rule=num_promotions_rule_min)

    return model


def simulate_period(customers_orders, cust_behaviour, fix_order_costs, margin, cost_per_prom, max_proms, min_proms, period, feature_set, strategy="random", model_clf="non", model_regr="non"):
    cust = customers_orders.merge(cust_behaviour, how='left', on=["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom"], suffixes=('_x', ''))
    if strategy == "optimal":          
        cust = calc_true_expected_profit_for_opt_strategy(cust, fix_order_costs, margin, cost_per_prom)
        cust = decide_prom_with_lp(cust, max_proms, min_proms) # or func.decide_prom(cust_opt)
    elif strategy == "estimated":
        cust = predict_cust_behaviour_with_clf_and_regr_model(model_clf, model_regr, feature_set, cust_behaviour, cust, fix_order_costs, margin, cost_per_prom)
        cust = decide_prom_with_lp(cust, max_proms, min_proms)
    elif strategy == "random":
        no_proms = cust.shape[0] - max_proms
        prom = np.array([1]*max_proms + [0]*no_proms)
        np.random.shuffle(prom)
        cust["send_prom"] = prom.tolist()
    
    orders = generate_orders(cust, margin, fix_order_costs, cost_per_prom)
    orders["period"] = period
    orders["profit"] = orders["size"] * margin - (orders["has_bought"] * fix_order_costs) - (orders["has_prom"] * cost_per_prom)
    return orders



def evaluate_period(orders_opt, orders_random_forest, orders_gtb, orders_linear, orders_random, period):
    period_eval = {
        "strategy": ["optimal", "linear", "random_forest", "gtb", "random"],
        "proms_sent": [orders_opt["has_prom"].sum(), orders_random_forest["has_prom"].sum(), orders_linear["has_prom"].sum(), orders_gtb["has_prom"].sum(), orders_random["has_prom"].sum()], 
        "buys": [orders_opt["has_bought"].sum(), orders_random_forest["has_bought"].sum(), orders_linear["has_bought"].sum(), orders_gtb["has_bought"].sum(), orders_random["has_bought"].sum()], 
        "orders_sum": [round(orders_opt["size"].sum(), 1), round(orders_random_forest["size"].sum(), 1), round(orders_linear["size"].sum(), 1), round(orders_gtb["size"].sum(), 1), round(orders_random["size"].sum(), 1)], 
        "profit": [round(orders_opt["profit"].sum(), 1), round(orders_random_forest["profit"].sum(), 1), round(orders_linear["profit"].sum(), 1), round(orders_gtb["profit"].sum(), 1), round(orders_random["profit"].sum(), 1)], 
    }
    period_eval = pd.DataFrame(period_eval)
    period_eval["period"] = period
    return period_eval

def full_log_likelihood(w, X, y):
    score = np.dot(X, w).reshape(1, X.shape[0]).astype(float)
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(w, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0]).astype(float)
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))

def self_confusion_matrix(test_orders_set, true, pred, label):
    tp = test_orders_set[(true == 1) & (pred == 1)].shape[0]
    tn = test_orders_set[(true == 0) & (pred == 0)].shape[0]
    fp = test_orders_set[(true == 0) & (pred == 1)].shape[0]
    fn = test_orders_set[(true == 1) & (pred == 0)].shape[0]

    print("%s        Vorhersagen " %(label))
    print("Beobachtungen    |   kauft    |    kauft nicht")
    print("hat gekauft      |   %s       |    %s " %(tp, fn))
    print("hat nicht gekauft|   %s       |    %s " %(fp, tn))

def get_kpis(orders, label):
    kpis = {
        "models": [label],
        "proms_sent": [orders["has_prom"].sum()],
        "buys": [orders["has_bought"].sum()],
        "orders_sum": [round(orders["size"].sum(), 1)],
        "profit": [round(orders["profit"].sum(), 1)],
        "exp_profit": [round(orders["exp_profit"].sum(), 1)]
    }
    return pd.DataFrame(kpis)

def export_cust_behaviour_for_ampl(cust_behaviour, fix_order_costs, margin, fullfilepath):
    cust_behaviour_prom = cust_behaviour[['t1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 'p_prom', 'size_prom']]
    cust_behaviour_prom["R"] = cust_behaviour_prom["size_prom"] * margin - fix_order_costs
    cust_behaviour_prom = cust_behaviour_prom.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_prom": "P", "R": "R"})
    cust_behaviour_prom["w"] = 1
    cust_behaviour_no_prom = cust_behaviour[['t1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 'p_no_prom', 'size_no_prom']]
    cust_behaviour_no_prom["R"] = cust_behaviour_no_prom["size_no_prom"] * margin - fix_order_costs
    cust_behaviour_no_prom = cust_behaviour_no_prom.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_no_prom": "P", "R": "R"})
    cust_behaviour_no_prom["w"] = 0
    cust_behaviour_ampl = pd.concat([cust_behaviour_prom, cust_behaviour_no_prom]) 
    cust_behaviour_ampl = cust_behaviour_ampl[["w1", "w2", "w3", "a1" ,"a2" ,"a3", "w", "P", "R"]]
    cust_behaviour_ampl.to_csv(fullfilepath, sep=' ', index=False)
    print(cust_behaviour_ampl)

def export_cust_behaviour_for_ampl_2(cust_behaviour, fix_order_costs, margin, fullfilepath): 
    cust_behaviour_prom = cust_behaviour[['t1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 'p_prom', 'size_prom']]
    cust_behaviour_prom["R"] = cust_behaviour_prom["size_prom"] * margin - fix_order_costs
    cust_behaviour_prom = cust_behaviour_prom.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_prom": "P", "R": "R"})
    cust_behaviour_prom["w"] = 1
    cust_behaviour_no_prom = cust_behaviour[['t1_buy', 't2_buy', 't3_buy', 't1_prom', 't2_prom', 't3_prom', 'p_no_prom', 'size_no_prom']]
    cust_behaviour_no_prom["R"] = cust_behaviour_no_prom["size_no_prom"] * margin - fix_order_costs
    cust_behaviour_no_prom = cust_behaviour_no_prom.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_no_prom": "P", "R": "R"})
    cust_behaviour_no_prom["w"] = 0
    cust_behaviour_ampl = pd.concat([cust_behaviour_prom, cust_behaviour_no_prom]) 
    cust_behaviour_ampl = cust_behaviour_ampl[["w1", "w2", "w3", "a1" ,"a2" ,"a3", "w", "P", "R"]]

    f = open(fullfilepath, "w")
    f.write("data;\n")
    f.write("param: P R :=\n")
    for row_tuple in cust_behaviour_ampl.itertuples():
        print(str(row_tuple.w1) + "," + str(row_tuple.w2) + "," + str(row_tuple.w3) + "," + str(row_tuple.a1) + "," + str(row_tuple.a2) + "," + str(row_tuple.a3) + "," + str(row_tuple.w) + "," + str(row_tuple.P) + "," + str(row_tuple.R))
        f.write(str(row_tuple.w1) + " " + str(row_tuple.w2) + " " + str(row_tuple.w3) + " " + str(row_tuple.a1) + " " + str(row_tuple.a2) + " " + str(row_tuple.a3) + " " + str(row_tuple.w) + " " + str(row_tuple.P) + " " + str(row_tuple.R) + "\n")
    f.write(";\n")
    f.close()
    
    print(cust_behaviour_ampl)

def write_performance_log(fullfilepath, run, prom, policy, start_time, last_time, period = 0, info = '', print_out = True, write_header = False):
    step_ms = int((time.time_ns() - last_time) / 1000000)
    time_ms = int((time.time_ns() - start_time) / 1000000)
    if print_out:
        print("--- run %s, promotions %s, %s, %s, %s, %s ---" % (run, prom, policy, info, time_ms, step_ms))
        
    f = open(fullfilepath, "a", newline='')
    if write_header:
        f.write("time_ms;diff_ms;run;proms;policy;period;info\r\n")
    f.write("%s;%s;%s;%s;%s;%s;%s\r\n" %(time_ms, step_ms, run, prom, policy, period, info))
    f.close()
    return time.time_ns()

def write_dp_performance_log(fullfilepath, start_time, last_time, c, cust_count, iteration, print_out = True, write_header = False, min_c = 0, max_c = 0):
    step_ms = int((time.time_ns() - last_time) / 1000000)
    time_ms = int((time.time_ns() - start_time) / 1000000)
    if print_out:
        print(f'Anzahl Kunden mit Werbung {cust_count} bei Kosten von {c} in Iteration {iteration}')
        
    f = open(fullfilepath, "a", newline='')
    if write_header:
        f.write("time_ms;diff_ms;c;cust_count;iteration;min_c;max_c\r\n")
        return time.time_ns()
    f.write("%s;%s;%s;%s;%s;%s;%s \r\n" %(time_ms, step_ms, c, cust_count, iteration, min_c, max_c))
    f.close()
    return time.time_ns()