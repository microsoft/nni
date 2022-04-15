from copy import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from random import randint
from typing import Optional
import time
import torch

import datasets
from importlib_metadata import metadata
import numpy as np
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import argparse
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.hubert_compression_utils import HubertCompressModule
from hubert_utils import *
from sparta.common.utils import export_tesa
import copy
from shape_hook import ShapeHook

def measure_time(model, dummy_input, runtimes=200):
    times = []
    with torch.no_grad():
        for runtime in range(runtimes):
            torch.cuda.synchronize()
            start = time.time()
            out=model(*dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the training audio paths and labels."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to "
            "'validation'"
        },
    )
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    label_column_name: Optional[str] = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_length_seconds: Optional[float] = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_mask: Optional[bool] = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    finegrained: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If prune the model in the fine-grained way"
        }
    )
    coarsegrained: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If prune the model in the coarse-grained way"
        }
    )
    sparsity_ratio: Optional[float] = field(
        default=0,
        metadata={
            "help": "The sparsity ratio"
        }
    )
    onnx_name:  Optional[str] =  field(
        default='',
        metadata={
            "help": "onnx mode path"
        }
    )
    tesa_path:  Optional[str] =  field(
        default=None,
        metadata={
            "help": "onnx mode path"
        }
    )
    export_tesa: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if export onnx with tesa"
        }
    )
if __name__ == '__main__':
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            pass

    # Initialize our dataset and prepare it for the audio classification task.
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, split=data_args.train_split_name
    )
    raw_datasets["eval"] = load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, split=data_args.eval_split_name
    )

    if data_args.audio_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name {data_args.audio_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if data_args.label_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column_name]:
            wav = random_subsample(
                audio["array"], max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
            )
            output_batch["input_values"].append(wav)
        output_batch["labels"] = [label for label in batch[data_args.label_column_name]]

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column_name]:
            wav = audio["array"]
            output_batch["input_values"].append(wav)
        output_batch["labels"] = [label for label in batch[data_args.label_column_name]]

        return output_batch

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = raw_datasets["train"].features[data_args.label_column_name].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions[0], axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    dummy_input = torch.load('dummy_input.pth')
    hubert_dummy_input = torch.rand(32, 16000)
    model = model.to('cpu')
    model.config.return_dict=False
    # model.hubert.encoder.pos_conv_embed.conv = torch.nn.utils.remove_weight_norm(model.hubert.encoder.pos_conv_embed.conv)
    # setattr(model.config, 'use_return_dict', False)
    data = (dummy_input['input_values'].to('cpu'), dummy_input['attention_mask'].to('cpu'))

    
    # from ModelVisual2 import ModelVisual
    # mv = ModelVisual(model, data[:1])
    # mv.visualize('hubert_arch')
    # exit(-1)
    
    mask_file = 'checkpoints/finegrained/hubert_finegrained_mask.pth'

    mask = torch.load(mask_file)

    # import ipdb; ipdb.set_trace()
    
    ms =  ModelSpeedup(model, data[:1], mask_file, break_points=[])
    # ms =  ModelSpeedup(model.hubert, hubert_dummy_input , hubert_mask_file, break_points=['hubert.encoder.layers.0.attention.aten::view.191'])
    # import ipdb; ipdb.set_trace()

    propagated_mask = ms.propagate_mask()
    # ms.speedup_model()
    

    model(data[0])
    model.load_state_dict(torch.load('checkpoints/finegrained/hubert_finegrained_state_dict.pth',  map_location=torch.device('cpu')))
    
    # import ipdb; ipdb.set_trace()


    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    

    optimizer = trainer.create_optimizer()
    #dummy_input = torch.load('dummy_input.pth')
    device = torch.device('cpu')
    dummy_input = torch.rand((32, 16000)).to(device)
    #data = (dummy_input['input_values'].to('cuda'), dummy_input['attention_mask'].to('cuda'))
    model.eval().to(device)
    # apply_mask(model, propagated_mask, device)
    # Evaluation
    # import ipdb; ipdb.set_trace()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print('Accuracy:', metrics)
    # tmp_model = copy.deepcopy(model)
    os.makedirs('artifact_hubert_finegrained_onnx_with_tesa', exist_ok=True)
    sh = ShapeHook(model, data[:1])
    sh.export('artifact_hubert_finegrained_onnx_with_tesa/shape.json')
    export_tesa(model, data[:1], 'artifact_hubert_finegrained_onnx_with_tesa', propagated_mask)
    
    # import ipdb; ipdb.set_trace