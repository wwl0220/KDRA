import os
from functools import partial
import numpy as np
import pandas as pd
import json
import time
import argparse

import torch
import torch.optim.lr_scheduler as lr_scheduler

from meta.averagevaluemeter import AverageValueMeter
from meta.data_ensemble_process_mlp import load_train_val_data, load_test_data
from meta.base_model_selection_mlp import select_base_model
import meta.log as log

from meta.proto_net.model import load_protonet, evaluate
from meta.proto_net.engine import Engine


parser = argparse.ArgumentParser()

parser.add_argument('--new_attack', default='ftp')  # 'ftp', 'hulk', 'slowbody', 'http_post', 'cc', 'http_flood'
parser.add_argument('--n_model', default=7)  # number of base models
parser.add_argument('--data_shot', default=20)  # number of new attack samples 20, 15, 10, 5

parser.add_argument('--data_train_episodes', default=42)  # training episodes in meta-training
parser.add_argument('--data_cuda', default=True)

parser.add_argument('--model_hid_dim', default=16)
parser.add_argument('--model_z_dim', default=16)

parser.add_argument('--train_decay_every', default=20)
parser.add_argument('--train_patience', default=2000)
parser.add_argument('--train_epochs', default=200)

parser.add_argument('--seeds', default=42, type=int)

args = parser.parse_args()

train_model_path = 'results/best_train_model.pt'
test_model_path = 'results/best_test_model.pt'
test_proto_path = 'results/best_test_proto.pt'

log_exp_dir = 'results'
log_fields = 'loss,acc'

train_trace_file = os.path.join(log_exp_dir, 'train_trace.txt')
test_trace_file = os.path.join(log_exp_dir, 'test_trace.txt')

log_fields = log_fields.split(',')

engine = Engine()
meters = {'train': {field: AverageValueMeter() for field in log_fields},
          'val': {field: AverageValueMeter() for field in log_fields}}


def on_start(state):
    if state['train_flag']:
        if os.path.isfile(train_trace_file):
            os.remove(train_trace_file)
    else:
        if os.path.isfile(test_trace_file):
            os.remove(test_trace_file)
    state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], args.train_decay_every, gamma=0.1)


engine.hooks['on_start'] = on_start


def on_start_epoch(state):
    for split, split_meters in meters.items():
        for field, meter in split_meters.items():
            meter.reset()
    # state['scheduler'].step()  # pytorch 0.4.1


engine.hooks['on_start_epoch'] = on_start_epoch


def on_update(state):
    for field, meter in meters['train'].items():
        meter.add(state['output'][field])


engine.hooks['on_update'] = on_update


def on_end_epoch(hook_state, state):
    # np.random.seed(45)
    if 'best_loss' not in hook_state:
        hook_state['best_loss'] = np.inf
    if 'wait' not in hook_state:
        hook_state['wait'] = 0

    evaluate(state['model'],
             state['val_data'],
             meters['val'],
             args.data_cuda)

    meter_vals = log.extract_meter_values(meters)
    print("Epoch {:02d}: {:s}".format(state['epoch'], log.render_meter_values(meter_vals)))
    meter_vals['epoch'] = state['epoch']
    if state['train_flag']:
        with open(train_trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')
    else:
        with open(test_trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

    if meter_vals['val']['loss'] < hook_state['best_loss']:
        hook_state['best_loss'] = meter_vals['val']['loss']
        print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

        state['model'].cpu()
        if state['train_flag']:
            torch.save(state['model'], os.path.join(log_exp_dir, 'best_train_model.pt'))
            state['best_model_epoch'] = state['epoch']
            train_end_time = time.time()
            state['best_model_time'] = train_end_time - state['start_time']
        else:
            torch.save(state['model'], os.path.join(log_exp_dir, 'best_test_model.pt'))
            state['best_model_epoch'] = state['epoch']
            test_end_time = time.time()
            state['best_model_time'] = test_end_time - state['start_time']
            print("==> best proto (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))
            state['proto'].cpu()
            torch.save(state['proto'], os.path.join(log_exp_dir, 'best_test_proto.pt'))
            if args.data_cuda:
                state['proto'].cuda()

        if args.data_cuda:
            state['model'].cuda()

        hook_state['wait'] = 0
    else:
        hook_state['wait'] += 1

        if hook_state['wait'] > args.train_patience:
            print("==> patience {:d} exceeded".format(args.train_patience))
            state['stop'] = True


engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})


col_results = ['n_model', 'acc', 'dr', 'fr', 'train_best_model_epoch', 'train_best_model_time', 'test_best_model_epoch',
               'test_best_model_time', 'attack_models_chosen']

model_x_dim = 2 * args.n_model
attack_models_chosen = select_base_model(args.n_model, args.new_attack, args.data_shot, args.seeds)
print(attack_models_chosen)

torch.manual_seed(args.seeds)
if args.data_cuda:
    torch.cuda.manual_seed(args.seeds)
torch.cuda.empty_cache()

# meta-training
model = load_protonet(model_x_dim, args.model_hid_dim, args.model_z_dim)
if args.data_cuda:
    model.cuda()

train_data, val_data = load_train_val_data(attack_models_chosen, args.data_train_episodes, args.data_shot, args.seeds)
train_data = train_data.values()
val_data = val_data.values()

train_start_time = time.time()
train_best_model_epoch, train_best_model_time = engine.train(
    model=model,
    train_data=train_data,
    val_data=val_data,
    max_epoch=args.train_epochs,
    train_flag=True,
    start_time=train_start_time,
    is_cuda=args.data_cuda
)
engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

# meta-testing
best_train_model = torch.load(train_model_path)
if args.data_cuda:
    best_train_model.cuda()

test_train_data, test_val_data, test_eval_data = load_test_data(args.new_attack, attack_models_chosen, args.data_shot, args.seeds)
test_train_data = test_train_data.values()
test_val_data = test_val_data.values()

test_start_time = time.time()
test_best_model_epoch, test_best_model_time = engine.train(
    model=best_train_model,
    train_data=test_train_data,
    val_data=test_val_data,
    max_epoch=args.train_epochs,
    train_flag=False,
    start_time=test_start_time,
    is_cuda=args.data_cuda
)
engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

# test
best_test_model = torch.load(test_model_path)
if args.data_cuda:
    best_test_model.cuda()
best_test_proto = torch.load(test_proto_path)
if args.data_cuda:
    best_test_proto.cuda()

y_hat, target, output = best_test_model.test_predict(test_eval_data, best_test_proto, args.data_cuda)

true_attack = target[0]
true_benign = target[1]
predict_attack = y_hat[0]
predict_benign = y_hat[1]

acc = output['acc']
dr = torch.eq(predict_attack, true_attack).float().mean().item()
fr = torch.eq(predict_benign, true_benign).float().mean()
fr = 1.0 - fr.item()

print(output)

data_results = [[args.n_model, acc, dr, fr, train_best_model_epoch, train_best_model_time,
                 test_best_model_epoch, test_best_model_time, attack_models_chosen]]
df_results = pd.DataFrame(data_results, columns=col_results)
print(acc)
print(dr)
print(fr)

df_results.to_csv('results/kdra_'+args.new_attack+'_k_'+str(args.data_shot)+'_mlp.csv', index=False)
