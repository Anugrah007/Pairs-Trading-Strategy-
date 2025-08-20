# Parameter Optimization
import numpy as np
from .riskmeasure import RiskMeasure
from mango.tuner import Tuner

class bayesianOpt:

    def __init__(self,risk_measure):
        self.optimize_results = None
        self.risk_measure = risk_measure
       

    def optimize(self,beta,spread,S1,S2,param_grid,conf_dict):
        """
        Optimization Function for classifier with data inputs and scoring function
        """
        def objective(args_list):
            objs = []
            for params in args_list:

                window1 = params["window1"]
                window2 = params["window2"]
                sell_threshold = params["sell_threshold"]
                buy_threshold = params["buy_threshold"]
                clear_threshold = params["clear_threshold"]
                money = bayesianOpt.trade(S1,S2,spread,beta,window1,window2,sell_threshold,buy_threshold,clear_threshold)
                obj = self.risk_measure.calculate(money)
                objs.append(obj)
            return objs

        tuner_user = Tuner(param_grid, objective, conf_dict)
        optimize_results = tuner_user.maximize()
        self.optimize_results = optimize_results
        return optimize_results
    

    @staticmethod
    def trade(S1, S2,spread,beta, window1, window2,sell_threshold,buy_threshold,clear_threshold):
        

           # If window length is 0, exit
        if (window1 == 0) or (window2 == 0):
            return 0

        # Compute rolling mean and rolling standard deviation
        ma1 = spread.rolling(window=window1, center=False).mean()
        ma2 = spread.rolling(window=window2, center=False).mean()
        std = spread.rolling(window=window2, center=False).std()

        # Avoid division by zero in z-score calculation
        # std.replace(0, float("nan"), inplace=True)
        zscore = (ma1 - ma2) / std

        # Initialize trading simulation variables
        money_arr = []
        money = 1
        countS1 = 0
        countS2 = 0

        # Simulate trading
        for i in range(len(spread)-1):
        
            # Sell short if the z-score is above the sell threshold
            if zscore.iloc[i] > sell_threshold:
                money += S1.iloc[i] - S2.iloc[i] * beta
                countS1 -= 1
                countS2 += beta

            # Buy long if the z-score is below the buy threshold
            elif zscore.iloc[i] < buy_threshold:
                money -= S1.iloc[i] - S2.iloc[i] * beta
                countS1 += 1
                countS2 -= beta

            # Clear positions if the z-score is within the neutral range
            elif abs(zscore.iloc[i]) < clear_threshold:
                money += countS1 * S1.iloc[i] + countS2 * S2.iloc[i]
                countS1 = 0
                countS2 = 0

            money_arr.append(money)

        return money_arr
           



