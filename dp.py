import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functions as func
import time


def p(w1, w2, w3, a1, a2, a3, w, cust_behaviour):
    boolean_mask = (cust_behaviour["w1"] == w1) & (cust_behaviour["w2"] == w2) & (cust_behaviour["w3"] == w3) & (cust_behaviour["a1"] == a1) & (cust_behaviour["a2"] == a2) & (cust_behaviour["a3"] == a3)
    p = cust_behaviour.loc[boolean_mask, "p_"+str(int(w))]
    return float(p)


def r(w1, w2, w3, a1, a2, a3, w, fix_order_costs, margin, cust_behaviour):
    boolean_mask = (cust_behaviour["w1"] == w1) & (cust_behaviour["w2"] == w2) & (cust_behaviour["w3"] == w3) & (cust_behaviour["a1"] == a1) & (cust_behaviour["a2"] == a2) & (cust_behaviour["a3"] == a3)
    r = cust_behaviour.loc[boolean_mask, "r_"+str(int(w))]
    return float(float(r) * margin - fix_order_costs)


def q(w1, w2, w3, a1, a2, a3, w, i, gamma, start_value, I, c, fix_order_costs, margin, cust_behaviour, value_table):
    q = (-c*w + p(w1, w2, w3, a1, a2, a3, w, cust_behaviour) * (r(w1, w2, w3, a1, a2, a3, w, fix_order_costs, margin, cust_behaviour) + gamma * v_i(w2, w3, w, a2, a3, 1, i, start_value, I, value_table)) + (1-p(w1, w2, w3, a1, a2, a3, w, cust_behaviour)) * gamma * v_i(w2, w3, w, a2, a3, 0, i, start_value, I, value_table))
    return float(q)


def v_i(w1, w2, w3, a1, a2, a3, i, start_value, I, value_table):
    if i >= I:
        return start_value
    boolean_mask = (value_table["w1"] == w1) & (value_table["w2"] == w2) & (value_table["w3"] == w3) & (value_table["a1"] == a1) & (value_table["a2"] == a2) & (value_table["a3"] == a3)
    v_i = value_table.loc[boolean_mask, "V_"+str(int(i+1))]
    return float(v_i)


def update_v(w1, w2, w3, a1, a2, a3, q_0, q_1, w, v, i, value_table):
    boolean_mask = (value_table["w1"] == w1) & (value_table["w2"] == w2) & (value_table["w3"] == w3) & (value_table["a1"] == a1) & (value_table["a2"] == a2) & (value_table["a3"] == a3)
    value_table.loc[boolean_mask, "Q_0_"+str(int(i))] = float(q_0)
    value_table.loc[boolean_mask, "Q_1_"+str(int(i))] = float(q_1)
    value_table.loc[boolean_mask, "w_"+str(int(i))] = int(w)
    value_table.loc[boolean_mask, "V_"+str(int(i))] = float(v)
    return value_table


def get_mean_value_diff(i, I, value_table):
    if i >= I:
        return 999
    return (value_table["V_"+str(int(i))] - value_table["V_"+str(int(i+1))]).mean()


def init_value_table(I, cust_behaviour, start_value):    
    value_table = cust_behaviour.copy()

    for i in range(1, I+1):
        new_cols = pd.DataFrame([[start_value, start_value, start_value, 0]], columns=["V_"+str(i), "Q_0_"+str(i), "Q_1_"+str(i), "w_"+str(i)])
        value_table = pd.concat([value_table, new_cols], axis=1)
    return value_table


def v(I, gamma, start_value, c, fix_order_costs, margin, cust_behaviour, value_table):
    for i in range(I, 0, -1):
        for index, row in cust_behaviour.iterrows():
            q_1 = q(row["w1"], row["w2"], row["w3"], row["a1"], row["a2"], row["a3"], 1, i, gamma, start_value, I, c, fix_order_costs, margin, cust_behaviour, value_table)
            q_0 = q(row["w1"], row["w2"], row["w3"], row["a1"], row["a2"], row["a3"], 0, i, gamma, start_value, I, c, fix_order_costs, margin, cust_behaviour, value_table)
            v = max(float(q_1), float(q_0))
            w = 1 if (q_1 > q_0) else 0
            value_table = update_v(row["w1"], row["w2"], row["w3"], row["a1"], row["a2"], row["a3"], q_0, q_1, w, v, i, value_table)
        mean_value_diff = get_mean_value_diff(i, I, value_table)
        if mean_value_diff <= 0.001:
            break
    #print("Last Iteration: "+str(i)+" "+str(mean_value_diff))
    cust_behaviour["V"] = value_table["V_"+str(i)]
    cust_behaviour["Q_0"] = value_table["Q_0_"+str(i)]
    cust_behaviour["Q_1"] = value_table["Q_1_"+str(i)]
    cust_behaviour["w"] = value_table["w_"+str(i)]
    return cust_behaviour


def decide_with_dp(cust_behaviour, cust, budget, fullfilepath):
    cust_behaviour = cust_behaviour.rename(columns={"t1_buy": "a1", "t2_buy": "a2", "t3_buy": "a3", "t1_prom": "w1", "t2_prom": "w2", "t3_prom": "w3", "p_no_prom": "p_0", "size_no_prom": "r_0", "p_prom": "p_1", "size_prom": "r_1"})
    start_value = 1000.0001
    I = 50 # I = 50
    fix_order_costs = 9.0
    margin = 0.5
    gamma = 0.995  # Diskontfaktor   
    c = 1.0
    value_table = init_value_table(I, cust_behaviour, start_value)
    start_time = time.time_ns()
    last_time = start_time
    
    last_time = func.write_dp_performance_log(fullfilepath, start_time, last_time, 0, 0, 0, print_out = True, write_header = True)
    
    
    iteration = 0
    min_c = 0.1
    max_c = 100.0
    c = min_c
    while True:
        iteration = iteration + 1
        current_cust = cust.copy()
        cust_behaviour = v(I, gamma, start_value, c, fix_order_costs, margin, cust_behaviour, value_table)
        current_cust = current_cust.merge(cust_behaviour, how='left', left_on=["t1_buy", "t2_buy", "t3_buy", "t1_prom", "t2_prom", "t3_prom"], right_on=["a1", "a2", "a3", "w1", "w2", "w3"], suffixes=('_x', ''))
        cust_count = current_cust["w"].sum()
        last_time = func.write_dp_performance_log(fullfilepath, start_time, last_time, c, cust_count, iteration, True, False)
            
        if abs(int(cust_count) - int(budget)) < 5:
            current_cust = current_cust.rename(columns={"w": "send_prom", "p_0": "p_no_prom", "r_0": "size_no_prom", "p_1": "p_prom", "r_1": "size_prom"})
            current_cust = current_cust.sort_values("send_prom", ascending=False, ignore_index=True)
            current_cust["send_prom"] = 0
            current_cust.loc[0:(int(budget)-1), "send_prom"] = 1
            print("Min C "+str(min_c)+" Max C "+str(max_c)+" Kosten "+str(c))
            return current_cust
        elif cust_count > budget:
            min_c = c
            c = max_c - ((max_c - min_c) / 2.0)
        elif cust_count < budget:
            max_c = c
            c = max_c - ((max_c - min_c) / 2.0)
        if (max_c - min_c) < 0.01 or iteration > 10:
            
            current_cust = current_cust.rename(columns={"w": "send_prom", "p_0": "p_no_prom", "r_0": "size_no_prom", "p_1": "p_prom", "r_1": "size_prom"})
            current_cust = current_cust.sort_values("send_prom", ascending=False, ignore_index=True)
            current_cust["send_prom"] = 0
            current_cust.loc[0:(int(budget)-1), "send_prom"] = 1
            print("Min C "+str(min_c)+" Max C "+str(max_c)+" Kosten "+str(c))
            return current_cust