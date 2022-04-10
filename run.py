"""
main running file 
"""

# load packages 
import os 
import sys
import numpy as np 
import pandas as pd 

# load files
from src.model_train import *
from src.model_eval import *
from src.model_pred import *

# ======= modules ======

def run_evaluation():
    """ train eval split to select best params and models"""
    model = LogisticRegressionCustom
    # model = RandomForestClassifierCustom
    # model = LightGBMClassifierCustom
    # model = XGBoostClassifierCustom
    # params = {'max_depth': 100, 'n_estimator': 300} # , 'reg_lambda': 0.05}
    params = {}
    run_model_evaluation(model, params)

def run_prediction():
    """ final predictions """
    model = ''  # TODO: ...
    params = {} # TODO: ...
    final_predict(model, params)

# ====== main ======

def main(targets):
    """ main running """
    if 'eval' in targets:
        run_evaluation()

    if 'pred' in targets:
        run_prediction()
    
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)