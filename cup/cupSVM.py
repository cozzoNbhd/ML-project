from keras.layers import CategoryEncoding
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split, ParameterSampler, learning_curve 
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, make_scorer
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import loguniform 
from cupUtilities import DatasetProcessor
from joblib import parallel_backend
import tensorflow.keras.backend as K
import os
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import seaborn as sns
from scipy import stats
import time





def mean_euclidean_error2(y_true, y_pred):
    #ensure inputs are NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    
    #y_true and y_pred must be 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    

    #compute Euclidean distance for each sample
    euclidean_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    #compute and return the mean of the distances
    return np.mean(euclidean_distances)


def plot_learning_curves(estimator, X, y, cv, scoring, target_name, bp):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), 
        random_state=42
    )
    
    #compute mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    #plot della learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='green')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='green')
    
    plt.title(f"Learning Curve for {target_name} with params {bp}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score (Negative MEE)")
    plt.legend(loc="best")
    plt.grid()
    
    
    filename = f"learning_curve_{target_name}.png"
    plt.savefig(filename)
    print(f"Learning curve saved as {filename}")
    plt.close()


def grid_search_kfold():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DatasetProcessor(ROOT_DIR)
    
    #read dataset
    x_data, y_data = processor.read_tr(split=False)  
    x_ts = processor.read_ts()  
    scaler = MinMaxScaler()
    #normalize
    
    x_data = scaler.fit_transform(x_data)
    x_ts = scaler.transform(x_ts)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    models = {}
    cv_figures = {} 
    mee_scores = {}  

    
    mee_scorer = make_scorer(mean_euclidean_error2, greater_is_better=False)
    final_predictions = []
    target_names = ['TARGET_x', 'TARGET_y', 'TARGET_z']
    
    for i, target in enumerate(target_names):
        y_train_target = y_train[:, i]
        y_test_target = y_test[:, i]
        

        param_grid = {
            'C': [15, 20, 50],
            'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
            'gamma': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5,0.7, 0.8],
            'epsilon': [0.01, 0.07, 0.1],
            'degree' : [2,3,4]
        }
        
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        model = SVR()
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=mee_scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        print(f"Optimizing {target}...")
        grid_search.fit(x_train, y_train_target)
        print(f"Best parameters for {target}: {grid_search.best_params_}")
        
        best_model = grid_search.best_estimator_
        models[target] = best_model
        #test set predictions
        pred_test = best_model.predict(x_ts)

        final_predictions.append(pred_test)
        #plot learning curve
        print(f"Plotting learning curve for {target}...")
        plot_learning_curves(
            estimator=best_model,
            X=x_train,
            y=y_train_target,
            cv=inner_cv,
            scoring=mee_scorer,
            target_name=target,
            bp=grid_search.best_params_
        )
        
        #test set predictions
        pred_test = best_model.predict(x_ts)
       
    
    train_predictions = np.vstack([
        models['TARGET_x'].predict(x_train),
        models['TARGET_y'].predict(x_train),
        models['TARGET_z'].predict(x_train)
    ]).T

    val_predictions = np.vstack([
        models['TARGET_x'].predict(x_test),
        models['TARGET_y'].predict(x_test),
        models['TARGET_z'].predict(x_test)
    ]).T

    test_predictions = np.vstack([
        models['TARGET_x'].predict(x_ts),
        models['TARGET_y'].predict(x_ts),
        models['TARGET_z'].predict(x_ts)
    ]).T

    #compute MEE scores
    mee_train = mean_euclidean_error2(y_train, train_predictions)
    mee_val = mean_euclidean_error2(y_test, val_predictions)
    mee_scores = {
        'train': mee_train,
        'validation': mee_val,
    }
    
    print(f"MEE (Training): {mee_train}")
    print(f"MEE (Validation): {mee_val}")

    #save blind test
    final_predictions = np.array(final_predictions).T
    processor.write_blind_results(final_predictions)
    print("Results saved successfully!")
    
    return models, cv_figures, mee_scores


def main():
   grid_search_kfold()
    
    

if __name__ == "__main__":
    main()