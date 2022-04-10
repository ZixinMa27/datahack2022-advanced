"""
engineer features 
"""

# load packages 
import os
import re
import numpy as np
import pandas as pd 

import scipy.sparse as sparse

from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer

# specify paths 
RAW_PATH = 'data/raw'
FEATURE_PATH_VAL = 'data/preprocessed_val'
FEATURE_PATH_TEST = 'data/preprocessed_test'

# local copy of functions
polarity_score = SentimentIntensityAnalyzer().polarity_scores

# ========== features ==========
# store features to FEATURE_PATH 
def common_words_embeddings(
    df: pd.DataFrame,
    use_stemmer: bool=True,
    num_grams = (1, 1),
    ignore_digits: bool=True,
    N: int = 2000,
    count_vectorizer = None,
    return_count_vectorizer: bool = True
):
    """ 
    extract the sparse common words for each sentence

    :param df: the train/test dataset 
    :param use_stemmer: True to use stemmer
    :param num_grams: tuple for ngrams
    :param N: the top N words 
    :param count_vectorizer if using the same 
    :return a dataframe of n by N, where n is the number of samples
    """
    # if use stemmers 
    if use_stemmer:
        stemmer = PorterStemmer()
        sentence = df['Sentence'].apply(
            lambda x: ' '.join([stemmer.stem(word) for word in x.split()])
        )
    else:
        sentence = df['Sentence']
    
    if ignore_digits:
        sentence = sentence.apply(lambda x: ''.join([y for y in x if not y.isdigit()]))
        sentence = sentence.apply(lambda x: re.sub(r"[^\w\s]", '', x) if x.isdigit() else x)
    
    # if using a trained count vectorizer 
    if count_vectorizer is None:
        vectorizer = CountVectorizer(
            max_features=N, 
            stop_words='english',
            ngram_range=num_grams
        )
        X = vectorizer.fit_transform(sentence)
        # vectorizer.get_feature_names()
        embeddings = X.toarray()
    else: 
        embeddings = count_vectorizer.transform(sentence).toarray()
    
    # return 
    if return_count_vectorizer: # return a new one 
        return embeddings, vectorizer
    else: 
        return embeddings


def sentiment_scores_embeddings(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    compute the sentiment scores of each sentence 

    :param df: the dataframe
    :return the sentiment scores
    """
    sentiment_scores_list = df['Sentence'].apply(lambda x: polarity_score(x)).to_list()
    sentiment_scores = pd.DataFrame(sentiment_scores_list)
    return sentiment_scores

# ======== y values ========
def convert_y_values(df):
    """ cast y to int """
    y_int = df[['Sentiment']].replace({'negative': -1, 'neutral': 0, 'positive': 1})
    return y_int

# ======== concat features =========
def make_features(train_df, test_df, is_test: bool=False):
    """
    create and save train test or train val features 

    :param is_test: True if for testing, and false for validation
    """
    # ---- create features ----
    # sentiment scores 
    train_sentiment_scores = sentiment_scores_embeddings(train_df)
    test_sentiment_scores = sentiment_scores_embeddings(test_df)

    # count vectorizer 
    train_common_words_1_1, cv_1_1 = common_words_embeddings(train_df, use_stemmer=True, num_grams=(1, 1), N=3000, return_count_vectorizer=True)
    train_common_words_1_2, cv_1_2 = common_words_embeddings(train_df, use_stemmer=True, num_grams=(1, 2), N=3000, return_count_vectorizer=True)
    train_common_words_2_2, cv_2_2 = common_words_embeddings(train_df, use_stemmer=True, num_grams=(2, 2), N=2000, return_count_vectorizer=True)
    train_common_words_3_3, cv_3_3 = common_words_embeddings(train_df, use_stemmer=True, num_grams=(3, 3), N=1000, return_count_vectorizer=True)
    test_common_words_1_1 = common_words_embeddings(test_df, use_stemmer=True,  count_vectorizer=cv_1_1, return_count_vectorizer=False)
    test_common_words_1_2 = common_words_embeddings(test_df, use_stemmer=True, count_vectorizer=cv_1_2, return_count_vectorizer=False)
    test_common_words_2_2 = common_words_embeddings(test_df, use_stemmer=True, count_vectorizer=cv_2_2, return_count_vectorizer=False)
    test_common_words_3_3 = common_words_embeddings(test_df, use_stemmer=True, count_vectorizer=cv_3_3, return_count_vectorizer=False)
    # TODO: maybe add more 


    # ---- convert y ----
    y_train = convert_y_values(train_df)
    if not is_test:
        y_val = convert_y_values(test_df)
    

    # ---- save features ----
    # sentiment scores 
    save_dense_embeddings(train_sentiment_scores, 'train_sentiment_scores', is_test=is_test)
    save_dense_embeddings(test_sentiment_scores, 'test_sentiment_scores', is_test=is_test)

    # count vectorizer 
    save_sparse_embeddings(train_common_words_1_1, 'train_cv_1_1', is_test=is_test)
    save_sparse_embeddings(train_common_words_1_2, 'train_cv_1_2', is_test=is_test)
    save_sparse_embeddings(train_common_words_2_2, 'train_cv_2_2', is_test=is_test)
    save_sparse_embeddings(train_common_words_3_3, 'train_cv_3_3', is_test=is_test)
    save_sparse_embeddings(test_common_words_1_1, 'test_cv_1_1', is_test=is_test)
    save_sparse_embeddings(test_common_words_1_2, 'test_cv_1_2', is_test=is_test)
    save_sparse_embeddings(test_common_words_2_2, 'test_cv_2_2', is_test=is_test)
    save_sparse_embeddings(test_common_words_3_3, 'test_cv_3_3', is_test=is_test)

    # y values 
    save_dense_embeddings(y_train, 'y_train', is_test=is_test)
    if not is_test:
        save_dense_embeddings(y_val, 'y_test', is_test=is_test)


def create_X_y(is_test: bool=False):
    """
    create 
    - X_train, y_train, X_val, y_val if is_test is False 
    - X_train, y_train, X_test if is_test is True 
    """
    # ---- read features ----
    train_sentiment_scores = read_dense_embeddings('train_sentiment_scores', is_test=is_test)
    test_sentiment_scores = read_dense_embeddings('test_sentiment_scores', is_test=is_test)

    train_cv_1_1 = read_sparse_embeddings('train_cv_1_1', is_test=is_test)
    train_cv_1_2 = read_sparse_embeddings('train_cv_1_2', is_test=is_test)
    train_cv_2_2 = read_sparse_embeddings('train_cv_2_2', is_test=is_test)
    train_cv_3_3 = read_sparse_embeddings('train_cv_3_3', is_test=is_test)
    test_cv_1_1 = read_sparse_embeddings('test_cv_1_1', is_test=is_test)
    test_cv_1_2 = read_sparse_embeddings('test_cv_1_2', is_test=is_test)
    test_cv_2_2 = read_sparse_embeddings('test_cv_2_2', is_test=is_test)
    test_cv_3_3 = read_sparse_embeddings('test_cv_3_3', is_test=is_test)

    # print(train_sentiment_scores)
    # print(train_cv_1_1)
    # ---- concat ----
    X_train = sparse.hstack([
        train_sentiment_scores.to_numpy(),
        train_cv_1_1, train_cv_1_2, train_cv_2_2, train_cv_3_3
    ])
    # X_train = sparse.csr_matrix(X_train)
    X_test = sparse.hstack([
        test_sentiment_scores.to_numpy(),
        test_cv_1_1, test_cv_1_2, test_cv_2_2, test_cv_3_3
    ])
    # X_test = sparse.csr_matrix(X_test)

    # ---- read y values -----
    y_train = read_dense_embeddings('y_train', is_test=is_test).squeeze()
    if not is_test:
        y_val = read_dense_embeddings('y_test', is_test=is_test).squeeze()
    
    # return 
    if not is_test:
        return X_train, y_train, X_test, y_val
    else:
        return X_train, y_train, X_test


# ----------------------------
# ======== feature IO ========
# ----------------------------

def save_dense_embeddings(data: pd.DataFrame, name: str, is_test: bool=False):
    """ store dense embeddings """
    path = FEATURE_PATH_TEST if is_test else FEATURE_PATH_VAL
    data.to_parquet(os.path.join(path, name))

def save_sparse_embeddings(data: pd.DataFrame or np.ndarray, name:str, is_test: bool=False):
    """ store sparse embeddings """
    # convert to sparse 
    data_sparse = sparse.csr_matrix(data)
    path = FEATURE_PATH_TEST if is_test else FEATURE_PATH_VAL
    sparse.save_npz(os.path.join(path, name + '.npz'), data_sparse)

def read_dense_embeddings(name: str, is_test: bool=False) -> pd.DataFrame:
    """ load dense embeddings """
    path = FEATURE_PATH_TEST if is_test else FEATURE_PATH_VAL
    df = pd.read_parquet(os.path.join(path, name))
    return df 

def read_sparse_embeddings(name: str, is_test: bool=False) -> sparse.csr_matrix:
    """ load sparse embeddings """
    path = FEATURE_PATH_TEST if is_test else FEATURE_PATH_VAL
    matrix = sparse.load_npz(os.path.join(path, name + '.npz'))
    return matrix

