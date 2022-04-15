import os
import re

def get_acc(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        acc = list(filter(lambda x : 'Accuracy:' in x, lines))
        return acc[0].strip()

if __name__ == '__main__':
    prefix =  '../checkpoints'
    models = ['bert', 'mobilenet', 'hubert']
    patterns = ['coarse', 'finegrained']
    
    for m in models:
        for p in patterns:
            fpath=f'../checkpoints/{m}/{p}.log'
            if os.path.exists(fpath):
                acc = get_acc(fpath)
                print(f'Model:{m} Pattern:{p} {acc}')