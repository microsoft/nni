import os
import logging
import random
import math
import time
import argparse
import distutils.util
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from macro import GeneralNetwork
from micro import MicroNetwork
from nni.nas.pytorch.fixed import apply_fixed_architecture
import datasets
from utils import accuracy, reward_accuracy
from nni.nas.pytorch.utils import AverageMeterGroup, to_device
import json

def set_random_seed(seed):
    # logger.info("set random seed for data reading: {}".format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed_all(seed)
    if FLAGS.is_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# parser args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_output_dir",
        type=distutils.util.strtobool,
        default=True,
        help="Whether to clean the output dir if existed. (default: %(default)s)")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Directory containing the dataset and embedding file. (default: %(default)s)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_macro",
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
        "--global_seed",
        type=int,
        default=1234,
        help="Seed for reproduction. (default: %(default)s)")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="The number of training epochs. (default: %(default)s)")
    parser.add_argument("--retrain_for", choices=["macro", "micro"], default="macro")
    parser.add_argument(
        "--child_grad_bound",
        type=float,
        default=5.0,
        help="The threshold for gradient clipping. (default: %(default)s)")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="The initial learning rate. (default: %(default)s)")
    parser.add_argument(
        "--child_l2_reg",
        type=float,
        default=3e-6,
        help="Weight decay factor. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_max",
        type=float,
        default=0.02,
        help="The max learning rate. (default: %(default)s)")
    parser.add_argument(
        "--child_lr_min",
        type=float,
        default=0.001,
        help="The min learning rate. (default: %(default)s)")
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
        "--checkpoint_dir",
        type=str,
        default="best_checkpoint",
        help="Path for saved checkpoints. (default: %(default)s)")
    parser.add_argument(
        "--is_cuda",
        type=distutils.util.strtobool,
        default=True,
        help="Specify the device type. (default: %(default)s)")
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

def eval_once(child_model, device, eval_set, criterion, valid_dataloader=None, test_dataloader=None):
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
            x, y = batch
            x, y = to_device(x, device), to_device(y, device)
            logits = child_model(x)

            if isinstance(logits, tuple):
                logits, aux_logits = logits
                aux_loss = criterion(aux_logits, y)
            else:
                aux_loss = 0.

            loss = criterion(logits, y)
            loss = loss + aux_weight * aux_loss
            #             loss = loss.mean()
            preds = logits.argmax(dim=1).long()
            acc = torch.eq(preds, y.long()).long().sum().item()

            losses.append(loss)
            tot_acc += acc
            tot += len(y)

    losses = torch.tensor(losses)
    loss = losses.mean()
    if tot > 0:
        final_acc = float(tot_acc) / tot
    else:
        final_acc = 0
        logger.info("Error in calculating final_acc")
    return final_acc, loss

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

def train(child_model, device, output_dir='./output'):
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename='./retrain.log',
                        filemode='a')
    logger = logging.getLogger("nni.enas-retrain")

    workers = 4
    aux_weight = 0.4
    batch_size = FLAGS.batch_size #128
    lr = FLAGS.lr # 0.02
    log_every = FLAGS.log_every  #50
    output_dir = FLAGS.output_dir #'./output_macro'
    num_epochs = FLAGS.num_epochs
    child_grad_bound = FLAGS.child_grad_bound #5.0
    child_l2_reg = FLAGS.child_l2_reg #3e-6
    child_lr_max = FLAGS.child_lr_max #0.002
    child_lr_min = FLAGS.child_lr_min #0.001
    child_lr_decay_scheme = FLAGS.child_lr_decay_scheme
    child_lr_T_0 = FLAGS.child_lr_T_0 # 10
    child_lr_T_mul = FLAGS.child_lr_T_mul #2

    logger.info("Build dataloader")
    dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    n_train = len(dataset_train)
    split = n_train // 10
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=workers)
    valid_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   sampler=valid_sampler,
                                                   num_workers=workers)
    test_dataloader = torch.utils.data.DataLoader(dataset_valid,
                                                  batch_size=batch_size,
                                                  num_workers=workers)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(child_model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4, nesterov=True)
    # optimizer = optim.Adam(child_model.parameters(), eps=1e-3, weight_decay=FLAGS.child_l2_reg)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    child_model.to(device)
    criterion.to(device)


    logger.info('Start training')
    start_time = time.time()
    step = 0

    # save path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_save_path = os.path.join(output_dir, "model.pth")
    best_model_save_path = os.path.join(output_dir, "best_model.pth")
    best_acc = 0
    start_epoch = 0

    # load model
    if FLAGS.load_checkpoint:
        print('** Load model **')
        logger.info('** Load model **')
        child_model.load_state_dict(torch.load(best_model_save_path)['child_model_state_dict'])

    acc_l = []

    # train
    for epoch in range(start_epoch, num_epochs):
        lr = update_lr(optimizer,
            epoch,
            l2_reg= child_l2_reg, #1e-4,
            lr_warmup_val=None,
            lr_init=lr,
            lr_decay_scheme=child_lr_decay_scheme,
            lr_max=child_lr_max,#0.05,
            lr_min=child_lr_min,#0.001,
            lr_T_0=child_lr_T_0,#10,
            lr_T_mul=child_lr_T_mul)#2)
        child_model.train()
        for batch in train_dataloader:
            step += 1

            x, y = batch
            x, y = to_device(x, device), to_device(y, device)
            logits = child_model(x)

            if isinstance(logits, tuple):
                logits, aux_logits = logits
                aux_loss = criterion(aux_logits, y)
            else:
                aux_loss = 0.

            acc = accuracy(logits, y)
            loss = criterion(logits, y)
            loss = loss + aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = 0
            trainable_params = child_model.parameters()

            for param in trainable_params:
                grad_norm = nn.utils.clip_grad_norm_(param, child_grad_bound)  # clip grad
                param = grad_norm

            optimizer.step()

            if step % log_every == 0:
                curr_time = time.time()
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += "ch_step={:<6d}".format(step)
                log_string += " loss={:<8.6f}".format(loss)
                log_string += " lr={:<8.4f}".format(lr)
                log_string += " |g|={:<8.4f}".format(grad_norm)
                log_string += " tr_acc={:<8.4f}/{:>3d}".format(acc['acc1'], logits.size()[0])
                log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
                logger.info(log_string)

        epoch += 1
        save_state = {
            'step': step,
            'epoch': epoch,
            'child_model_state_dict': child_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        #         print(' Epoch {:<3d} loss: {:<.2f} '.format(epoch, loss))
        torch.save(save_state, model_save_path)
        child_model.eval()

        if epoch % FLAGS.eval_every_epochs ==0:
            logger.info("Epoch {}: Eval".format(epoch))
            eval_acc, eval_loss = eval_once(child_model, device, "test", criterion, test_dataloader=test_dataloader)
            logger.info(
                "ch_step={} {}_accuracy={:<6.4f} {}_loss={:<6.4f}".format(step, "test", eval_acc, "test", eval_loss))
            if eval_acc > best_acc:
                best_acc = eval_acc
                logger.info("Save best model")
                save_state = {
                    'step': step,
                    'epoch': epoch,
                    'child_model_state_dict': child_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
            torch.save(save_state, best_model_save_path)

            result['accuracy'].append('Epoch {} acc: {:<6.4f}'.format(epoch, eval_acc,))
            acc_l.append(eval_acc)

        print(result['accuracy'][-1])

    print('max acc %.3f at epoch: %i'%(max(acc_l), np.argmax(np.array(acc_l))))
    print('Time cost: %.4f hours'%( float(time.time() - start_time) /3600.  ))
    return result

aux_weight = 0.4
result = {'accuracy':[],}
def dump_global_result(res_path,global_result, sort_keys = False):
    with open(res_path, "w") as ss_file:
        json.dump(global_result, ss_file, sort_keys=sort_keys, indent=2)

def main():
    parse_args()

    device = torch.device("cuda" if FLAGS.is_cuda else "cpu")

    set_random_seed(FLAGS.global_seed)

    # macro search
    if FLAGS.retrain_for == 'macro':
        print('** Macro search **')
        child_model = GeneralNetwork()
        apply_fixed_architecture(child_model, './macro.json')

    # micro search
    elif FLAGS.retrain_for == 'micro':
        print(' ** Micro search **')
        child_model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1, use_aux_heads=True)
        apply_fixed_architecture(child_model, './micro.json')

    train(child_model, device)

if __name__ == "__main__":
    main()

