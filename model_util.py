import numpy as np
import copy
import scipy as scipy
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import discriminant_analysis
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from joblib import parallel_backend
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB, CategoricalNB
import datetime
import time
import joblib
from joblib import load
import inspect
import xgboost as xgb
import os

# Const
############################################### 

DEFAULT_BOOTSTRAP_N = 10000 


# Get test and training sets
############################################### 

# Current possible train/test splits: 'train_test_split' (unbalanced), 'undersample_1:3_train_test'
# Current glaucoma definitions: 'Glaucoma_Binary_all', 'Glaucoma_Binary_diagnosed'
# y_test / y_train returns labels replaced with "0" for Control and "1" for Glaucoma

def get_train_test_datasets(df, split_definition, glaucoma_definition, features):
    # Exclude those with NaN split_definition or glaucoma_definition
    eligibile_df = df[(df[split_definition].isna()==False) & (df[glaucoma_definition].isna()==False)]
    
    train_set = eligibile_df[eligibile_df[split_definition]=='train']
    test_set = eligibile_df[eligibile_df[split_definition]=='test']

    X_train = train_set[features]
    y_train = train_set[glaucoma_definition]

    X_test = test_set[features]
    y_test = test_set[glaucoma_definition]

    # New
    y_train = y_train.map(pd.Series({
        "Control": 0,
        "Glaucoma": 1,
    }))

    y_test = y_test.map(pd.Series({
        "Control": 0,
        "Glaucoma": 1,
    }))

    # Added reset index ??? if buggy
    
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# Model evaluation
############################################### 

### Bootstrapping

# Combined
def get_bootstrapped_measures(y_test, test_pr_prob, t, **kwargs):
    def evalf(data_1, data_2):
        #t = get_ideal_threshold(data_1, data_2)
        predictions = (data_2 >= t).astype(int)

        auroc = metrics.roc_auc_score(data_1, data_2)
        precision, recall, pr_t = metrics.precision_recall_curve(data_1, data_2, pos_label=1, drop_intermediate=False)
        auprc = metrics.auc(recall, precision)

        sensitivity, specificity, ppv, npv, f1_score = get_confusion_matrix_measures(data_1, data_2, t=t)

        return auroc, auprc, sensitivity, specificity, ppv, npv, f1_score

    n_bootstrap_resamples = kwargs.get('n_bootstrap_resamples', DEFAULT_BOOTSTRAP_N)
    res = stats.bootstrap(data=(y_test, test_pr_prob), statistic=evalf, n_resamples=n_bootstrap_resamples, paired=True, random_state=np.random.default_rng(2024), method='percentile')
    return res


# Get Youden index threshold cutoff for best sensitivity & specificity
def get_ideal_threshold(y, probabilities):
    fpr_arr, tpr_arr, trial_t_arr = metrics.roc_curve(y, probabilities, pos_label=1, drop_intermediate=False)

    # WORKING:
    best_t = trial_t_arr[np.argmax(tpr_arr - fpr_arr)]

    # # TEST: Fix at level of 70% specificity
    # fixed_spec_80_i = np.abs(fpr_arr - 0.30).argmin()
    # best_t = trial_t_arr[fixed_spec_80_i]
    
    return best_t


def get_confusion_matrix_measures(y, probabilities, t=None):
    if t == None:
        # Find best threshold based on Youden
        t = get_ideal_threshold(y, probabilities)
    predictions = (probabilities >= t).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    f1_score = (2 * ppv * sensitivity) / (ppv + sensitivity)

    return sensitivity, specificity, ppv, npv, f1_score

def get_estimator_probabilities(estimator, X):
    is_svc = (estimator.__class__.__name__ == 'LinearSVC') | (estimator.__class__.__name__ == 'SVC')
    is_mlp = estimator.__class__.__name__ == 'MLP'
    if is_svc == True:
        confidence_function = estimator.decision_function
    else:
        confidence_function = estimator.predict_proba

    probabilities = confidence_function(X)
    if is_svc == False and is_mlp == False:
        probabilities = probabilities[:, 1]

    return probabilities

def get_scoring_metrics(estimator, X, y):
    #predictions = estimator.predict(X)
    probabilities = get_estimator_probabilities(estimator, X)

    # Threshold via best sens and spec method
    best_t = get_ideal_threshold(y, probabilities)
    best_t_predictions = (probabilities >= best_t).astype(int)
    roc_auc = metrics.roc_auc_score(y, probabilities)
    sensitivity, specificity, ppv, npv, f1_score = get_confusion_matrix_measures(y, probabilities, t=best_t)

    precision, recall, pr_t = metrics.precision_recall_curve(y, probabilities, pos_label=1, drop_intermediate=False)
    pr_auc = metrics.auc(recall, precision)

    return {
        'roc_auc': roc_auc,
        'optimal_threshold': best_t,
        'sensitivity_recall': sensitivity,
        'specificity': specificity,
        'ppv_precision': ppv,
        'npv': npv,
        'f1_score': f1_score,
        'pr_auc': pr_auc,
    }






