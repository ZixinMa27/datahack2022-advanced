"""
visualize information
"""

# load packages 
import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# import file 
from src.feature_engineering import * 

# specify path
FIGURE_PATH = 'fig'
FEATURE_PATH_TEST = 'data/preprocessed_test'

# ======== EDA part =======
def visualize_sentiment_proportion(train_df: pd.DataFrame):
    """ visualize distribution of train sentiment """
    train_df['Sentiment'].value_counts(normalize=True).plot(kind='bar')
    plt.ylabel('Proportion')
    plt.xlabel('Sentiment')
    plt.title('Proportion of Sentiment in Train')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(FIGURE_PATH, 'sentiment_proportion.png'), dpi=800)

def sentiment_scores_by_sentiment(sentiment_intensity_name):
    """ read from preprocessed dataset """
    # read train_sentiment scores
    train_sentiment_scores = read_dense_embeddings('train_sentiment_scores', is_test=True)
    y_train = read_dense_embeddings('y_train', is_test=True).squeeze()
    
    # visualize 
    fig, ax = plt.subplots()
    sentiment_labels = [-1, 0, 1]
    sentiment_names = ['negative', 'neutral', 'positive']
    for sentiment_name in sentiment_labels:
        subset_df = train_sentiment_scores[y_train == sentiment_name]
        subset_df[sentiment_intensity_name].plot(kind='kde', ax=ax)

    plt.legend(sentiment_names)
    plt.title(sentiment_intensity_name + ' Intensity by Label')
    plt.savefig(os.path.join(FIGURE_PATH, f'sentiment_intensity_by_label_{sentiment_intensity_name}.png'), dpi=800)


def uni_count(train_df: pd.DataFrame):
    top = common_words_embeddings(
        train_df,
        True,
        (1, 1),
        True,
        10,
        None,
        True
        )
    name = top[1].get_feature_names()
    count_list = top[0].sum(axis=0)
    combine = dict(zip(name, count_list))

    lists =sorted(combine.items(), key=lambda kv: kv[1], reverse=True)
    dicts = dict(lists)
    plt.bar(*zip(*dicts.items()),width = 0.5)
    plt.ylabel("count")
    plt.xlabel("Unigram")
    plt.title("Top 10 Unigram Count")
    plt.xticks(rotation=45)
    
    plt.savefig(os.path.join(FIGURE_PATH, 'unigram_count.png'), dpi=800)

def bi_count(train_df: pd.DataFrame):
    top = common_words_embeddings(
        train_df,
        True,
        (2, 2),
        True,
        10,
        None,
        True
        )
    name = top[1].get_feature_names()
    count_list = top[0].sum(axis=0)
    combine = dict(zip(name, count_list))

    lists =sorted(combine.items(), key=lambda kv: kv[1], reverse=True)
    dicts = dict(lists)
    plt.bar(*zip(*dicts.items()),width = 0.5)
    plt.ylabel("count")
    plt.xlabel("Bigram")
    plt.title("Top 10 Bigram Count")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(FIGURE_PATH, 'bigram_count.png'), dpi=800)

def tri_count(train_df: pd.DataFrame):
    top = common_words_embeddings(
        train_df,
        True,
        (3, 3),
        True,
        10,
        None,
        True
        )
    name = top[1].get_feature_names()
    count_list = top[0].sum(axis=0)
    combine = dict(zip(name, count_list))

    lists =sorted(combine.items(), key=lambda kv: kv[1], reverse=True)
    dicts = dict(lists)
    plt.bar(*zip(*dicts.items()),width = 0.5)
    plt.ylabel("count")
    plt.xlabel("Trigram")
    plt.title("Top 10 Trigram Count")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(FIGURE_PATH, 'trigram_count.png'), dpi=800)


