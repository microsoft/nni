import json
import os
import argparse
import tqdm

from .model import db, NlpTrialConfig, NlpTrialStats, NlpIntermediateStats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Path to extracted NLP data dir.')
    args = parser.parse_args()
    with db, tqdm.tqdm(total=len(os.listdir(args.input_dir)), desc="creating tables") as pbar:
        db.create_tables([NlpTrialConfig, NlpTrialStats, NlpIntermediateStats])
        json_files = os.listdir(args.input_dir)
        for json_file in json_files:
            pbar.update(1)
            if json_file.endswith('.json'):
                log_path = os.path.join(args.input_dir, json_file)
                cur = json.load(open(log_path, 'r'))
                arch = json.loads(cur['recepie'])
                unested_arch = {}
                for k in arch.keys():
                    # print(k)
                    unested_arch['{}_op'.format(k)] = arch[k]['op']
                    for i in range(len(arch[k]['input'])):
                        unested_arch['{}_input_{}'.format(k, i)] = arch[k]['input'][i]
                config = NlpTrialConfig.create(arch=unested_arch, dataset=cur['data'][5:])
                if cur['status'] == 'OK':
                    trial_stats = NlpTrialStats.create(config=config, train_loss=cur['train_losses'][-1], val_loss=cur['val_losses'][-1],
                                                       test_loss=cur['test_losses'][-1], training_time=cur['wall_times'][-1])
                    epochs = 50
                    intermediate_stats = []
                    for epoch in range(epochs):
                        epoch_res = {
                            'train_loss' : cur['train_losses'][epoch],
                            'val_loss' : cur['val_losses'][epoch],
                            'test_loss' : cur['test_losses'][epoch],
                            'training_time' : cur['wall_times'][epoch]
                        }
                        epoch_res.update(current_epoch=epoch + 1, trial=trial_stats)
                        intermediate_stats.append(epoch_res)
                    NlpIntermediateStats.insert_many(intermediate_stats).execute(db)


if __name__ == '__main__':
    main()
