import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint



def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs[(keys[i], keys[j])] = result
    return score_matrix, pvalue_matrix, pairs


def get_top_k_pairs(pairs,k):
    pairs_data = {key:value[1]  for (key, value) in pairs.items()}
    pairs_data = sorted(pairs_data.items(), key=lambda x: x[1])
    return pairs_data[0:k]


def get_cointergrated_coeff(y:pd.Series,x:pd.Series):

    """
    y = beta*x + e
    y-beta*x = e
    alpha = -beta


    Return: alpha
    """
    x = sm.add_constant(x)
    regress = sm.OLS(y,x)
    regress = regress.fit()
    alpha = -regress.params.iloc[-1]
    return alpha



def buy_signal_plots(spread,S1,S2,window1=60,window2=5,upper=1,lower=-1,label="Actual"):

    # Plot the ratios and buy and sell signals from z score
  
    ma1 = spread.rolling(window=window1, center=False).mean()
    ma2 = spread.rolling(window=window2, center=False).mean()
    std = spread.rolling(window=window2, center=False).std()
    zscore = ((ma1 - ma2)/std)

    # Compute the z score for each day
    zscore.name = 'z-score'

    plt.figure(figsize=(15,7))
    zscore.plot()

    plt.scatter(list(zscore[zscore>upper].index),zscore[zscore>upper],color="red")
    plt.scatter(list(zscore[zscore<lower].index),zscore[zscore<lower],color="green")  #
    plt.axhline(0, color='black')
    plt.axhline(upper, color='red', linestyle='--')
    plt.axhline(lower, color='green', linestyle='--')
    plt.legend([f'Rolling Spread {label} z-score',f'{upper}', f'{lower}','Mean'])
    plt.show()



def calculate_drawdown(money_arr):
    """
    Calculate the maximum drawdown of a money array.
    """
    # Replace zero with one to avoid division by zero
    money_arr = np.array(money_arr).astype(int)
    money_arr = np.where(money_arr == 0 , 1, money_arr)

    #Remove zero values
    cumulative_max = np.maximum.accumulate(money_arr)
    
    # Calculate the drawdown at each point in the array
    drawdowns = (money_arr - cumulative_max) / cumulative_max
   
    return drawdowns