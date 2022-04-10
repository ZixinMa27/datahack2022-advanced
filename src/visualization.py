"""
visualize informations
"""

# load packages 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# ======== EDA part =======
def visualize_sentiment_proportion(train_df: pd.DataFrame):
    """ viuslize distribution of train sentiment """
    train_df['Sentiment'].value_counts(normalize=True).plot(kind='bar')

