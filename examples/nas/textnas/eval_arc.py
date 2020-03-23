# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import shutil
import sys
import random
import math

import time
import datetime
import argparse
import distutils.util

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as Func

from macro_child import MacroChild
from dataloader import read_data_sst


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_output_dir",
        type=distutils.util.strtobool,
        default=True,
        help="Whether to clean the output dir if existed. (default: %(default)s)")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="glove",
        help="Embedding type. (default: %(default)s)")
    parser.add_argument(
        "--child_fixed_arc",
        type=str,
        required=True,
        help="Architecture description. (default: %(default)s)")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Directory containing the dataset and embedding file. (default: %(default)s)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_decay_scheme",
        type=str,
        default="cosine",
        help="Learning rate annealing strategy, only 'cosine' supported. (default: %(default)s)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of samples each batch for training. (default: %(default)s)")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Number of samples each batch for evaluation. (default: %(default)s)")
    parser.add_argument(
        "--class_num",
        type=int,
        default=5,
        help="The number of categories. (default: %(default)s)")
    parser.add_argument(
        "--global_seed",
        type=int,
        default=1234,
        help="Seed for reproduction. (default: %(default)s)")
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=64,
        help="The maximum length of the sentence. (default: %(default)s)")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="The number of training epochs. (default: %(default)s)")
    parser.add_argument(
        "--child_num_layers",
        type=int,
        default=24,
        help="The layer number of the architecture. (default: %(default)s)")
    parser.add_argument(
        "--child_out_filters",
        type=int,
        default=256,
        help="The dimension of hidden states. (default: %(default)s)")
    parser.add_argument(
        "--child_out_filters_scale",
        type=int,
        default=1,
        help="The scale of hidden state dimension. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_T_0",
        type=int,
        default=10,
        help="The length of one cycle. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_T_mul",
        type=int,
        default=2,
        help="The multiplication factor per cycle. (default: %(default)s)")
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="The threshold to cut off low frequent words. (default: %(default)s)")
    parser.add_argument(
        "--num_last_layer_output",
        type=int,
        default=0,
        help="The last n layers as output, 0 for all. (default: %(default)s)")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="The sample ratio for the training set. (default: %(default)s)")
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=1.0,
        help="The sample ratio for the dev set. (default: %(default)s)")
    parser.add_argument(
        "--child_grad_bound",
        type=float,
        default=5.0,
        help="The threshold for gradient clipping. (default: %(default)s)")
    parser.add_argument(
        "--child_lr",
        type=float,
        default=0.02,
        help="The initial learning rate. (default: %(default)s)")
    parser.add_argument(
        "--cnn_keep_prob",
        type=float,
        default=0.8,
        help="Keep prob for cnn layer. (default: %(default)s)")
    parser.add_argument(
        "--final_output_keep_prob",
        type=float,
        default=1.0,
        help="Keep prob for the last output layer. (default: %(default)s)")
    parser.add_argument(
        "--lstm_out_keep_prob",
        type=float,
        default=0.8,
        help="Keep prob for the RNN layer. (default: %(default)s)")
    parser.add_argument(
        "--embed_keep_prob",
        type=float,
        default=0.8,
        help="Keep prob for the embedding layer. (default: %(default)s)")
    parser.add_argument(
        "--attention_keep_prob",
        type=float,
        default=0.8,
        help="Keep prob for the self-attention layer. (default: %(default)s)")
    parser.add_argument(
        "--child_l2_reg",
        type=float,
        default=3e-6,
        help="Weight decay factor. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_max",
        type=float,
        default=0.002,
        help="The max learning rate. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_min",
        type=float,
        default=0.001,
        help="The min learning rate. (default: %(default)s)")
    parser.add_argument(
        "--child_optim_algo",
        type=str,
        default="adam",
        help="Optimization algorithm. (default: %(default)s)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="best_checkpoint",
        help="Path for saved checkpoints. (default: %(default)s)")
    parser.add_argument(
        "--output_type",
        type=str,
        default="avg_pool",
        help="Opertor type for the time steps reduction. (default: %(default)s)")
    parser.add_argument(
        "--multi_path",
        type=distutils.util.strtobool,
        default=False,
        help="Search for multiple path in the architecture. (default: %(default)s)")
    parser.add_argument(
        "--is_binary",
        type=distutils.util.strtobool,
        default=False,
        help="Binary label for sst dataset. (default: %(default)s)")
    parser.add_argument(
        "--all_layer_output",
        type=distutils.util.strtobool,
        default=True,
        help="Use all layers as output. (default: %(default)s)")
    parser.add_argument(
        "--output_linear_combine",
        type=distutils.util.strtobool,
        default=True,
        help="Combine all the layers in linear way. (default: %(default)s)")
    parser.add_argument(
        "--is_mask",
        type=distutils.util.strtobool,
        default=True,
        help="Apply mask. (default: %(default)s)")
    parser.add_argument(
        "--fixed_seed",
        type=distutils.util.strtobool,
        default=True,
        help="Fix the seed. (default: %(default)s)")
    parser.add_argument(
        "--load_checkpoint",
        type=distutils.util.strtobool,
        default=False,
        help="Wether to load checkpoint. (default: %(default)s)")
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="How many steps to log. (default: %(default)s)")
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="How many epochs to eval. (default: %(default)s)")

    global FLAGS

    FLAGS = parser.parse_args()


def set_random_seed(seed):
    print("-" * 80)
    print("set random seed for data reading: {}".format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(embedding, num_layers):
    print("num layers: {0}".format(num_layers))
    assert FLAGS.child_fixed_arc is not None, "Architecture should be provided."

    child_model = MacroChild(
        embedding=embedding,
        fixed_arc=FLAGS.child_fixed_arc,
        out_filters_scale=FLAGS.child_out_filters_scale,
        num_layers=num_layers,
        out_filters=FLAGS.child_out_filters,
        cnn_keep_prob=FLAGS.cnn_keep_prob,
        final_output_keep_prob=FLAGS.final_output_keep_prob,
        lstm_out_keep_prob=FLAGS.lstm_out_keep_prob,
        embed_keep_prob=FLAGS.embed_keep_prob,
        attention_keep_prob=FLAGS.attention_keep_prob,
        multi_path=FLAGS.multi_path,
        embedding_model=FLAGS.embedding_model,
        all_layer_output=FLAGS.all_layer_output,
        output_linear_combine=FLAGS.output_linear_combine,
        num_last_layer_output=FLAGS.num_last_layer_output,
        is_mask=FLAGS.is_mask,
        output_type=FLAGS.output_type,
        class_num=FLAGS.class_num)

    return child_model


def print_arc(arc, num_layers):
   start = 0
   for i in range(0, num_layers):
      end = start + i + 1
      if FLAGS.multi_path:
          end += 1
      out_str = "fixed_arc=\"$fixed_arc {0}\"".format(np.reshape(arc[start: end], [-1]))
      out_str = out_str.replace("[", "").replace("]", "")
      print(out_str)

      start = end


def eval_once(child_model, eval_set, criterion, valid_dataloader=None, test_dataloader=None):
    if eval_set == "test":
        assert test_dataloader is not None
        dataloader = test_dataloader
    elif eval_set == "valid":
        assert valid_dataloader is not None
        dataloader = valid_dataloader
    else:
        raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

    tot_acc = 0
    tot = 0
    losses = []

    with torch.no_grad():  # save memory
        for batch in dataloader:
            (sent_ids, mask), labels = batch

            sent_ids = sent_ids.cuda()
            mask = mask.cuda()
            labels = labels.cuda()

            logits = child_model(sent_ids, mask)  # run

            loss = criterion(logits, labels.long())
            loss = loss.mean()
            preds = logits.argmax(dim=1).long()
            acc = torch.eq(preds, labels.long()).long().sum().item()

            losses.append(loss)
            tot_acc += acc
            tot += len(labels)

    losses = torch.tensor(losses)
    loss = losses.mean()
    if tot > 0:
        final_acc = float(tot_acc) / tot
    else:
        final_acc = 0
        print("Error in calculating final_acc")
    return final_acc, loss


def print_user_flags(FLAGS, line_limit=80):
    print("-" * 80)

    log_strings = ""
    for flag_name in sorted(vars(FLAGS)):
        value = "{}".format(getattr(FLAGS, flag_name))
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        log_strings = log_strings + log_string
        log_strings = log_strings + "\n"
    print(log_strings)


def count_model_params(trainable_params):
    num_vars = 0
    for var in trainable_params:
        num_vars += np.prod([dim for dim in var.size()])
    return num_vars


def update_lr(
        optimizer,
        epoch,
        l2_reg=1e-4,
        lr_warmup_val=None,
        lr_init=0.1,
        lr_decay_scheme="cosine",
        lr_max=0.002,
        lr_min=0.000000001,
        lr_T_0=4,
        lr_T_mul=1,
        sync_replicas=False,
        num_aggregate=None,
        num_replicas=None):
    if lr_decay_scheme == "cosine":
        assert lr_max is not None, "Need lr_max to use lr_cosine"
        assert lr_min is not None, "Need lr_min to use lr_cosine"
        assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
        assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"

        T_i = lr_T_0
        t_epoch = epoch
        last_reset = 0
        while True:
            t_epoch -= T_i
            if t_epoch < 0:
              break
            last_reset += T_i
            T_i *= lr_T_mul

        T_curr = epoch - last_reset

        def _update():
            rate = T_curr / T_i * 3.1415926
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(rate))
            return lr

        learning_rate = _update()
    else:
        raise ValueError("Unknown learning rate decay scheme {}".format(lr_decay_scheme))

    #update lr in optimizer
    for params_group in optimizer.param_groups:
        params_group['lr'] = learning_rate
    return learning_rate

def train(data_path, output_dir, num_layers):
    print("Build dataloader")
    train_dataset, valid_dataset, test_dataset, embedding = \
        read_data_sst(data_path,
                      FLAGS.max_input_length,
                      FLAGS.min_count,
                      train_ratio=FLAGS.train_ratio,
                      valid_ratio=FLAGS.valid_ratio,
                      is_binary=FLAGS.is_binary)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.eval_batch_size, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=FLAGS.eval_batch_size, pin_memory=True)

    print("Build model")
    print("-" * 80)
    child_model = get_model(embedding, num_layers)
    print("Finish build model")

    for name, var in child_model.named_parameters():
        print(name, var.size(), var.requires_grad)  # output all params

    num_vars = count_model_params(child_model.parameters())
    print("Model has {} params".format(num_vars))

    for m in child_model.modules():  # initializer
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    criterion = nn.CrossEntropyLoss()

    # get optimizer
    if FLAGS.child_optim_algo == "adam":
        optimizer = optim.Adam(child_model.parameters(), eps=1e-3, weight_decay=FLAGS.child_l2_reg)  # with L2
    else:
        raise ValueError("Unknown optim_algo {}".format(FLAGS.child_optim_algo))

    child_model.cuda()
    criterion.cuda()

    fixed_arc = np.array([int(x) for x in FLAGS.child_fixed_arc.split(" ") if x])
    print_arc(fixed_arc, num_layers)

    print("Start training")
    print("-" * 80)
    start_time = time.time()
    step = 0

    # save path
    model_save_path = os.path.join(FLAGS.output_dir, "model.pth")
    best_model_save_path = os.path.join(FLAGS.output_dir, "best_model.pth")
    best_acc = 0
    start_epoch = 0
    if FLAGS.load_checkpoint:
        if os.path.isfile(model_save_path):
            checkpoint = torch.load(model_save_path, map_location = torch.device('cpu'))
            step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            child_model.load_state_dict(checkpoint['child_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(start_epoch, FLAGS.num_epochs):
        lr = update_lr(optimizer,
                       epoch,
                       l2_reg=FLAGS.child_l2_reg,
                       lr_warmup_val=None,
                       lr_init=FLAGS.child_lr,
                       lr_decay_scheme=FLAGS.child_lr_decay_scheme,
                       lr_max=FLAGS.child_lr_max,
                       lr_min=FLAGS.child_lr_min,
                       lr_T_0=FLAGS.child_lr_T_0,
                       lr_T_mul=FLAGS.child_lr_T_mul)
        child_model.train()
        for batch in train_dataloader:
            (sent_ids, mask), labels = batch

            sent_ids = sent_ids.cuda()
            mask = mask.cuda()
            labels = labels.cuda()

            step += 1

            logits = child_model(sent_ids, mask)  # run

            loss = criterion(logits, labels.long())
            loss = loss.mean()
            preds = logits.argmax(dim=1).long()
            acc = torch.eq(preds, labels.long()).long().sum().item()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = 0
            trainable_params = child_model.parameters()

            assert FLAGS.child_grad_bound is not None, "Need grad_bound to clip gradients."
            # compute the gradient norm value
            grad_norm = nn.utils.clip_grad_norm_(trainable_params, 99999999)
            for param in trainable_params:
                nn.utils.clip_grad_norm_(param, FLAGS.child_grad_bound)  # clip grad

            optimizer.step()

            if step % FLAGS.log_every == 0:
                curr_time = time.time()
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += "ch_step={:<6d}".format(step)
                log_string += " loss={:<8.6f}".format(loss)
                log_string += " lr={:<8.4f}".format(lr)
                log_string += " |g|={:<8.4f}".format(grad_norm)
                log_string += " tr_acc={:<3d}/{:>3d}".format(acc, logits.size()[0])
                log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
                print(log_string)

        epoch += 1
        save_state = {
            'step' : step,
            'epoch' : epoch,
            'child_model_state_dict' : child_model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()}
        torch.save(save_state, model_save_path)
        child_model.eval()
        print("Epoch {}: Eval".format(epoch))
        eval_acc, eval_loss = eval_once(child_model, "test", criterion, test_dataloader=test_dataloader)
        print("ch_step={} {}_accuracy={:<6.4f} {}_loss={:<6.4f}".format(step, "test", eval_acc, "test", eval_loss))
        if eval_acc > best_acc:
            best_acc = eval_acc
            print("Save best model")
            save_state = {
                'step' : step,
                'epoch' : epoch,
                'child_model_state_dict' : child_model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict()}
            torch.save(save_state, best_model_save_path)

    return eval_acc


def main():
    parse_args()
    print("-" * 80)
    if not os.path.isdir(FLAGS.output_dir):
        print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir, ignore_errors=True)
        os.makedirs(FLAGS.output_dir)
    print("-" * 80)
    log_file = os.path.join(FLAGS.output_dir, "stdout")
    print("Logging to {}".format(log_file))

    print_user_flags(FLAGS)

    if FLAGS.fixed_seed:
        set_random_seed(FLAGS.global_seed)

    train(FLAGS.data_path, FLAGS.output_dir, FLAGS.child_num_layers)


if __name__ == "__main__":
  main()
