# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sklearn
import time
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.utils import Timer
from amlb.results import save_predictions_to_file


arch_choices = [(16), (64), (128), (256),
                (16, 16), (64, 64), (128, 128), (256, 256),
                (16, 16, 16), (64, 64, 64), (128, 128, 128), (256, 256, 256),
                (256, 128, 64, 16), (128, 64, 16), (64, 16),
                (16, 64, 128, 256), (16, 64, 128), (16, 64)]

SEARCH_SPACE = {
    "hidden_layer_sizes": {"_type":"choice", "_value": arch_choices},
    "learning_rate_init": {"_type":"choice", "_value": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
    "alpha": {"_type":"choice", "_value": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]},
    "momentum": {"_type":"uniform","_value":[0, 1]},
    "beta_1": {"_type":"uniform","_value":[0, 1]},
    "tol": {"_type":"choice", "_value": [0.001, 0.0005, 0.0001, 0.00005, 0.00001]},
    "max_iter": {"_type":"randint", "_value": [2, 256]},
}

def preprocess_mlp(dataset, log):
    '''
    For MLP:
    - For numerical features, normalize them after null imputation. 
    - For categorical features, use one-hot encoding after null imputation. 
    '''
    cat_columns, num_columns = [], []
    shift_amount = 0
    for i, f in enumerate(dataset.features):
        if f.is_target:
            shift_amount += 1
            continue
        elif f.is_categorical():
            cat_columns.append(i - shift_amount)
        else:
            num_columns.append(i - shift_amount)

    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('onehot_encoder', OneHotEncoder()),
                             ])
    
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ('standard_scaler', StandardScaler()),
                             ])
    
    data_pipeline = ColumnTransformer([
        ('categorical', cat_pipeline, cat_columns),
        ('numerical', num_pipeline, num_columns),
    ])

    data_pipeline.fit(np.concatenate([dataset.train.X, dataset.test.X], axis=0))
    
    X_train = data_pipeline.transform(dataset.train.X)
    X_test = data_pipeline.transform(dataset.test.X)  
    
    return X_train, X_test

    
def run_mlp(dataset, config, tuner, log):
    """
    Using the given tuner, tune a random forest within the given time constraint.
    This function uses cross validation score as the feedback score to the tuner. 
    The search space on which tuners search on is defined above empirically as a global variable.
    """
    
    limit_type, trial_limit = config.framework_params['limit_type'], None
    if limit_type == 'ntrials':
        trial_limit = int(config.framework_params['trial_limit'])
    
    X_train, X_test = preprocess_mlp(dataset, log)
    y_train, y_test = dataset.train.y, dataset.test.y

    is_classification = config.type == 'classification'
    estimator = MLPClassifier if is_classification else MLPRegressor

    best_score, best_params, best_model = None, None, None
    score_higher_better = True

    tuner.update_search_space(SEARCH_SPACE)    
    
    start_time = time.time()
    trial_count = 0
    intermediate_scores = []
    intermediate_best_scores = []           # should be monotonically increasing 
    
    while True:
        try:            
            param_idx, cur_params = tuner.generate_parameters()
            if cur_params is not None and cur_params != {}:
                trial_count += 1
                train_params = cur_params.copy()
                
                if 'TRIAL_BUDGET' in cur_params:
                    train_params.pop('TRIAL_BUDGET')

                log.info("Trial {}: \n{}\n".format(param_idx, train_params))
                
                cur_model = estimator(random_state=config.seed, **train_params)
            
                # Here score is the output of score() from the estimator
                cur_score = cross_val_score(cur_model, X_train, y_train)
                cur_score = np.mean(cur_score)
                if np.isnan(cur_score):
                    cur_score = 0
            
                log.info("Score: {}\n".format(cur_score))
                if best_score is None or (score_higher_better and cur_score > best_score) or (not score_higher_better and cur_score < best_score):
                    best_score, best_params, best_model = cur_score, cur_params, cur_model    
            
                intermediate_scores.append(cur_score)
                intermediate_best_scores.append(best_score)
                tuner.receive_trial_result(param_idx, cur_params, cur_score)

            if limit_type == 'time':
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= config.max_runtime_seconds:
                    break
            elif limit_type == 'ntrials':
                if trial_count >= trial_limit:
                    break
        except:
            break

    # This line is required to fully terminate some advisors
    tuner.handle_terminate()
        
    log.info("Tuning done, the best parameters are:\n{}\n".format(best_params))

    # retrain on the whole dataset 
    with Timer() as training:
        best_model.fit(X_train, y_train)     
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test) if is_classification else None

    return probabilities, predictions, training, y_test, intermediate_scores, intermediate_best_scores
