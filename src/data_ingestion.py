"""
load data 
"""

# load packages 
import numpy as np
import pandas as pd 

# ======= io ========
def read_train():
    path = 'data/advanced_trainset.csv'
    df = pd.read_csv(path)
    return df 

def read_test():
    path = 'data/advanced_testset.csv'
    df = pd.read_csv(path)
    return df 
