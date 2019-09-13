import os
import glob
import numpy as np
import argparse

from os.path import join

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--all', action='store_true', help='Displays individual final results.')
parser.add_argument('--folder-path', type=str, default=None, help='The folder to evaluate if running in folder mode.')
parser.add_argument('--recursive', action='store_true', help='Apply folder-path mode to all sub-directories')


args = parser.parse_args()

if args.recursive:
    folders = [x[0] for x in os.walk(args.folder_path)]
else:
    folders = [args.folder_path if args.folder_path else './logs']


losses = []
accs = []
for folder in folders:
    losses = []
    accs = []
    for log_name in glob.iglob(join(folder, '*.log')):
        if not args.folder_path:
            losses = []
            accs = []
        arg = None
        with open(log_name) as f:
            for line in f:
                if 'Namespace' in line:
                    arg = line[19:-2]
                if not line.startswith('Test evaluation'): continue
                try:
                    loss = float(line[31:37])
                    acc = float(line[61:-3])/100
                except:
                    print('Could not convert number: {0}'.format(line[31:37]))

                losses.append(loss)
                accs.append(acc)
        if len(accs) == 0: continue

        acc_std = np.std(accs, ddof=1)
        acc_se = acc_std/np.sqrt(len(accs))

        loss_std = np.std(losses, ddof=1)
        loss_se = loss_std/np.sqrt(len(losses))


        if not args.folder_path:
            print('='*85)
            print('Test set results for log: {0}'.format(log_name))
            print('Arguments:\n{0}\n'.format(arg))
            print('Accuracy. Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(accs), acc_se, len(accs),
                np.mean(accs)-(1.96*acc_se), np.mean(accs)+(1.96*acc_se), np.median(accs)))
            print('Error.    Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(1.0-np.mean(accs), acc_se, len(accs),
                (1.0-np.mean(accs))-(1.96*acc_se), (1.0-np.mean(accs))+(1.96*acc_se), 1.0-np.median(accs)))
            print('Loss.     Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(losses), loss_se, len(losses),
                np.mean(losses)-(1.96*loss_se), np.mean(losses)+(1.96*loss_se), np.median(losses)))
            print('='*85)

            if args.all:
                print('Individual results:')
                for loss, acc in zip(losses, accs):
                    err = 1.0-acc
                    print('Loss: {0:.5f}, Accuracy: {1:.5f}, Error: {2:.5f}'.format(loss, acc, err))

    if args.folder_path:
        if len(accs) == 0:
            print('Test set results logs in folder {0} empty!'.format(folder))
            continue
        acc_std = np.std(accs, ddof=1)
        acc_se = acc_std/np.sqrt(len(accs))

        loss_std = np.std(losses, ddof=1)
        loss_se = loss_std/np.sqrt(len(losses))

        print('='*85)
        print('Test set results logs in folder: {0}'.format(folder))
        print('Arguments:\n{0}\n'.format(arg))
        print('Accuracy. Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(accs), acc_se, len(accs),
            np.mean(accs)-(1.96*acc_se), np.mean(accs)+(1.96*acc_se), np.median(accs)))
        print('Error.    Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(1.0-np.mean(accs), acc_se, len(accs),
            (1.0-np.mean(accs))-(1.96*acc_se), (1.0-np.mean(accs))+(1.96*acc_se), 1.0-np.median(accs)))
        print('Loss.     Median: {5:.5f}, Mean: {0:.5f}, Standard Error: {1:.5f}, Sample size: {2}, 95% CI: ({3:.5f},{4:.5f})'.format(np.mean(losses), loss_se, len(losses),
            np.mean(losses)-(1.96*loss_se), np.mean(losses)+(1.96*loss_se), np.median(losses)))
        print('='*85)

        if args.all:
            print('Individual results:')
            for loss, acc in zip(losses, accs):
                err = 1.0-acc
                print('Loss: {0:.5f}, Accuracy: {1:.5f}, Error: {2:.5f}'.format(loss, acc, err))




