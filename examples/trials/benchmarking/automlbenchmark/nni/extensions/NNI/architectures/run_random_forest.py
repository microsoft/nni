import logging
import sklearn
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.utils import Timer



SEARCH_SPACE = {
    "n_estimators": {"_type":"choice", "_value": [128, 256, 512, 1024, 2048, 4096]},
    "max_depth": {"_type":"choice", "_value": [5, 10, 25, 50, 100]},
}

    
def run_random_forest(dataset, config, tuner, log):
    is_classification = config.type == 'classification'

    X_train, X_test = impute(dataset.train.X, dataset.test.X)
    y_train, y_test = dataset.train.y, dataset.test.y

    estimator = RandomForestClassifier if is_classification else RandomForestRegressor

    best_score, best_params, best_model = None, None, None
    score_higher_better = True

    tuner.update_search_space(SEARCH_SPACE)
    start_time = time.time()
    while True:
        try:
            param_idx, cur_params = tuner.generate_parameters()
            if 'TRIAL_BUDGET' in cur_params:
                cur_params.pop('TRIAL_BUDGET')
            cur_model = estimator(random_state=config.seed, **cur_params)
            # Here score is the output of score() from the estimator
            cur_score = cross_val_score(cur_model, X_train, y_train)
            cur_score = sum(cur_score) / float(len(cur_score))
            if best_score is None or (score_higher_better and cur_score > best_score) or (not score_higher_better and cur_score < best_score):
                best_score, best_params, best_model = cur_score, cur_params, cur_model
                
            log.info("Trial {}: \n{}\nScore: {}\n".format(param_idx, cur_params, cur_score))
            tuner.receive_trial_result(param_idx, cur_params, cur_score)

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > config.max_runtime_seconds:
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

    return probabilities, predictions, training, y_test
