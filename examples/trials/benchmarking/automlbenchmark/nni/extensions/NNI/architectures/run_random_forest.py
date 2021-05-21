# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sklearn
import time
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.utils import Timer
from amlb.results import save_predictions_to_file


SEARCH_SPACE = {
    "n_estimators": {"_type":"randint", "_value": [8, 512]},
    "max_depth": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 256, 0]},   # 0 for None
    "min_samples_leaf": {"_type":"randint", "_value": [1, 8]},
    "min_samples_split": {"_type":"randint", "_value": [2, 16]},
    "max_leaf_nodes": {"_type":"randint", "_value": [0, 4096]}                    # 0 for None
}

SEARCH_SPACE_CHOICE = {
    "n_estimators": {"_type":"choice", "_value": [8, 16, 32, 64, 128, 256, 512]},
    "max_depth": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 0]},   # 0 for None
    "min_samples_leaf": {"_type":"choice", "_value": [1, 2, 4, 8]},
    "min_samples_split": {"_type":"choice", "_value": [2, 4, 8, 16]},
    "max_leaf_nodes": {"_type":"choice", "_value": [8, 32, 128, 512, 0]}     # 0 for None
}

SEARCH_SPACE_SIMPLE = {
    "n_estimators": {"_type":"choice", "_value": [10]},
    "max_depth": {"_type":"choice", "_value": [5]},
    "min_samples_leaf": {"_type":"choice", "_value": [8]},
    "min_samples_split": {"_type":"choice", "_value": [16]},
    "max_leaf_nodes": {"_type":"choice", "_value": [64]}
}


def preprocess_random_forest(dataset, log):
    '''
    For random forest:
    - Do nothing for numerical features except null imputation. 
    - For categorical features, use ordinal encoding to map them into integers. 
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
                             ('ordinal_encoder', OrdinalEncoder()),
                             ])
    
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ])
    
    data_pipeline = ColumnTransformer([
        ('categorical', cat_pipeline, cat_columns),
        ('numerical', num_pipeline, num_columns),
    ])

    data_pipeline.fit(np.concatenate([dataset.train.X, dataset.test.X], axis=0))
    
    X_train = data_pipeline.transform(dataset.train.X)
    X_test = data_pipeline.transform(dataset.test.X)  
    
    return X_train, X_test

    
def run_random_forest(dataset, config, tuner, log):
    """
    Using the given tuner, tune a random forest within the given time constraint.
    This function uses cross validation score as the feedback score to the tuner. 
    The search space on which tuners search on is defined above empirically as a global variable.
    """
    
    limit_type, trial_limit = config.framework_params['limit_type'], None
    if limit_type == 'ntrials':
        trial_limit = int(config.framework_params['trial_limit'])
    
    X_train, X_test = preprocess_random_forest(dataset, log)
    y_train, y_test = dataset.train.y, dataset.test.y

    is_classification = config.type == 'classification'
    estimator = RandomForestClassifier if is_classification else RandomForestRegressor

    best_score, best_params, best_model = None, None, None
    score_higher_better = True

    tuner.update_search_space(SEARCH_SPACE)    
    
    start_time = time.time()
    trial_count = 0
    intermediate_scores = []
    intermediate_best_scores = []           # should be monotonically increasing 
    
    while True:
        try:
            trial_count += 1
            param_idx, cur_params = tuner.generate_parameters()
            train_params = cur_params.copy()
            if 'TRIAL_BUDGET' in cur_params:
                train_params.pop('TRIAL_BUDGET')
            if cur_params['max_leaf_nodes'] == 0: 
                train_params.pop('max_leaf_nodes')
            if cur_params['max_depth'] == 0:
                train_params.pop('max_depth')
            log.info("Trial {}: \n{}\n".format(param_idx, cur_params))
                
            cur_model = estimator(random_state=config.seed, **train_params)
            
            # Here score is the output of score() from the estimator
            cur_score = cross_val_score(cur_model, X_train, y_train)
            cur_score = sum(cur_score) / float(len(cur_score))
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
