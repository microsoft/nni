import pandas as pd
import sys


if len(sys.argv) != 2:
    print("Usage: python parse_result_csv.py <result.csv file>")
    exit(0)

result_file = sys.argv[1]
result = pd.read_csv(result_file)

task_ids = result['id'].unique()

with open(result_file + '.parsed', 'w') as out_f:
    for task_id in task_ids:
        task_results = result[result['id'] == task_id]
        task_name = task_results.task.unique()[0]
        out_f.write("====================================================\n")
        out_f.write("Task ID: {}\n".format(task_id))
        out_f.write("Task Name: {}\n".format(task_name))
        folds = task_results['fold'].unique()
        for fold in folds:
            out_f.write("Fold {}:\n".format(fold))
            keep_parameters = ['framework', 'constraint', 'result', 'metric', 'params', 'utc', 'duration', 'acc', 'auc', 'logloss', 'r2', 'rmse']
            res = task_results[task_results['fold'] == fold][keep_parameters]
            out_f.write(res.to_string())
            out_f.write('\n')

        out_f.write('\n')   
