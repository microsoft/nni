import pandas as pd
import sys
import matplotlib.pyplot as plt


def generate_perf_report(result_file_name):
    result = pd.read_csv(result_file_name)
    task_ids = result['id'].unique()
    keep_parameters = ['framework', 'constraint', 'result', 'metric', 'params', 'utc', 'duration'] + list(result.columns[16:])    

    with open(result_file_name.replace('results.csv', 'reports/performances.txt'), 'w') as out_f:
        for task_id in task_ids:
            task_results = result[result['id'] == task_id]
            task_name = task_results.task.unique()[0]
            out_f.write("====================================================\n")
            out_f.write("Task ID: {}\n".format(task_id))
            out_f.write("Task Name: {}\n".format(task_name))
            folds = task_results['fold'].unique()
            for fold in folds:
                out_f.write("Fold {}:\n".format(fold))
                res = task_results[task_results['fold'] == fold][keep_parameters]
                out_f.write(res.to_string())
                out_f.write('\n')

            out_f.write('\n')   


def generate_convergence_report(result_file_name):
    result = pd.read_csv(result_file_name)
    scorelog_dir = result_file_name.replace('results.csv', 'scorelogs/')
    output_dir = result_file_name.replace('results.csv', 'reports/') 
    task_ids = result['id'].unique()
    for task_id in task_ids:
        task_results = result[result['id'] == task_id]
        task_name = task_results.task.unique()[0]
        folds = task_results['fold'].unique()
        # load scorelog files
        scores = []
        for fold in folds:            
            tuners = list(task_results[task_results.fold == fold]['framework'].unique())
            for tuner in tuners:
                scorelog_name = '{}_{}_{}.csv'.format(tuner.lower(), task_name, fold)
                intermediate_scores = pd.read_csv(scorelog_dir + scorelog_name)
                scores.append([tuner, fold, list(intermediate_scores['best_score'])])

        # generate a graph
        for tuner, fold, score in scores:
            plt.plot(score, label='{} Fold {}'.format(tuner, fold))
        plt.title(task_name)
        plt.xlabel("Number of Trials")
        plt.ylabel("Best Cross Validation Score")        
        plt.legend()
        plt.savefig(output_dir + '{}.jpg'.format(task_name))
        plt.close()

            
def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_result_csv.py <result.csv file>")
        exit(0)
    generate_perf_report(sys.argv[1])
    generate_convergence_report(sys.argv[1])
    

if __name__ == '__main__':
    main()
