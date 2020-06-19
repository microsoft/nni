import argparse
import numpy as np

from nasbench import api
from tqdm import tqdm

from .model import db, Nb101RunConfig, Nb101ComputedStats, Nb101IntermediateStats
from .graph_util import get_architecture_repr, hash_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the file to be converted, e.g., nasbench_full.tfrecord")
    args = parser.parse_args()
    print("Loading NAS-Bench-101 tfrecord...")
    nasbench = api.NASBench(args.input_file)
    with db:
        for hashval in tqdm(nasbench.hash_iterator(), desc="Dumping data into database"):
            metadata, metrics = nasbench.get_metrics_from_hash(hashval)
            architecture = get_architecture_repr(metadata["module_adjacency"], metadata["module_operations"])
            num_vertices = len(metadata["module_operations"])
            assert hashval == hash_module(architecture, num_vertices)
            for epochs in [4, 12, 36, 108]:
                run_config = Nb101RunConfig.create(arch=architecture, num_vertices=num_vertices,
                                                hash=hashval, num_epochs=epochs)

                for seed in range(3):
                    cur = metrics[epochs][seed]
                    run = Nb101ComputedStats.create(config=run_config,
                                                    train_acc=cur["final_train_accuracy"],
                                                    valid_acc=cur["final_validation_accuracy"],
                                                    test_acc=cur["final_test_accuracy"],
                                                    parameters=metadata["trainable_parameters"],
                                                    training_time=cur["final_training_time"])
                    for t in ["halfway", "final"]:
                        Nb101IntermediateStats.create(run=run,
                                                    current_epoch=epochs // 2 if t == "halfway" else epochs,
                                                    training_time=cur[t + "_training_time"],
                                                    train_acc=cur[t + "_train_accuracy"],
                                                    valid_acc=cur[t + "_validation_accuracy"],
                                                    test_acc=cur[t + "_test_accuracy"])
