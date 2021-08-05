import argparse

from tqdm import tqdm
from nasbench import api  # pylint: disable=import-error

from .model import db, Nb101TrialConfig, Nb101TrialStats, Nb101IntermediateStats
from .graph_util import nasbench_format_to_architecture_repr, hash_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='Path to the file to be converted, e.g., nasbench_full.tfrecord')
    args = parser.parse_args()
    nasbench = api.NASBench(args.input_file)
    with db:
        db.create_tables([Nb101TrialConfig, Nb101TrialStats, Nb101IntermediateStats])
        for hashval in tqdm(nasbench.hash_iterator(), desc='Dumping data into database'):
            metadata, metrics = nasbench.get_metrics_from_hash(hashval)
            num_vertices, architecture = nasbench_format_to_architecture_repr(
                metadata['module_adjacency'], metadata['module_operations'])
            assert hashval == hash_module(architecture, num_vertices)
            for epochs in [4, 12, 36, 108]:
                trial_config = Nb101TrialConfig.create(
                    arch=architecture,
                    num_vertices=num_vertices,
                    hash=hashval,
                    num_epochs=epochs
                )

                for seed in range(3):
                    cur = metrics[epochs][seed]
                    trial = Nb101TrialStats.create(
                        config=trial_config,
                        train_acc=cur['final_train_accuracy'] * 100,
                        valid_acc=cur['final_validation_accuracy'] * 100,
                        test_acc=cur['final_test_accuracy'] * 100,
                        parameters=metadata['trainable_parameters'] / 1e6,
                        training_time=cur['final_training_time'] * 60
                    )
                    for t in ['halfway', 'final']:
                        Nb101IntermediateStats.create(
                            trial=trial,
                            current_epoch=epochs // 2 if t == 'halfway' else epochs,
                            training_time=cur[t + '_training_time'],
                            train_acc=cur[t + '_train_accuracy'] * 100,
                            valid_acc=cur[t + '_validation_accuracy'] * 100,
                            test_acc=cur[t + '_test_accuracy'] * 100
                        )


if __name__ == '__main__':
    main()
