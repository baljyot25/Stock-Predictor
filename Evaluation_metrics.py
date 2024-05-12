import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,precision_score, recall_score, f1_score

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def MSE(y_true,y_pred):
    return mean_squared_error(y_true, y_pred)

def RMSE(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def R2(y_true,y_pred):
    return r2_score(y_true, y_pred)

def SMAPE(y_true,y_pred):
    return round( 
        np.mean( 
            np.abs(y_pred - y_true) / 
            ((np.abs(y_pred) + np.abs(y_true))/2) 
        )*100, 2
    )

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)
