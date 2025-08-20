

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd



class RiskMeasure(ABC):

    @abstractmethod
    def calculate(self, money_arr):
        pass


class CumulativeReturn(RiskMeasure):

    def __init__(self):
        pass

    def calculate(self, money_arr):
        return money_arr[-1]
    

class ExponationalUtility(RiskMeasure):

    def __init__(self, risk_aversion=0.01):
        self.risk_aversion = risk_aversion

    def calculate(self, money_arr):
        return -1*np.mean(np.exp(-self.risk_aversion *np.array(money_arr)))