# Created by xunannancy at 2021/9/25
"""
refer to codes for audio generation: https://github.com/vincentherrmann/pytorch-wavenet.git
"""
import spacetimeformer
from einops import rearrange, repeat
import torch.distributions as pyd
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import json
import os
import argparse
import yaml
from FNN import HistoryConcatLoader, HistoryConcatLoaderSpace
from utils import Pytorch_DNN_exp, Pytorch_DNN_validation, Pytorch_DNN_testing, print_network, merge_parameters, \
    run_evaluate_V3
import torch.nn.functional as F
import math
import time
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    """
    NOTE: comment original codes due to the following errors
    RuntimeError: Legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with static forward method. (Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)
    """
    pads = [0] * (input.ndim * 2)
    pads[2 * dimension + (1 if pad_start else 0)
         ] = target_size - input.shape[dimension]
    return torch.nn.functional.pad(input, pads[::-1], mode='constant', value=value)


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos %
                           dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(
            self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


# model = Spacetimeformer(
#     sliding_window=self.param_dict['sliding_window'],
#     external_features=self.config['exp_params']['external_features'],
#     history_column_names=self.dataloader.history_column_names,
#     target_val_column_names=self.dataloader.target_val_column_names,
#     normalization=self.param_dict['normalization'],
#     layers=self.param_dict['layers'],
#     blocks=self.param_dict['blocks'],
#     dilation_channels=self.param_dict['dilation_channels'],
#     residual_channels=self.param_dict['residual_channels'],
#     skip_channels=self.param_dict['skip_channels'],
#     end_channels=self.param_dict['end_channels'],
#     kernel_size=self.param_dict['kernel_size']
# )


class Spacetimeformer_exp(spacetimeformer.spacetimeformer_model.Spacetimeformer_Forecaster):
    def __init__(self, file, param_dict, config):
        super().__init__(d_x=6, d_yc=3, d_yt=3, max_seq_len=config['model_params']['context_points']+config['model_params']['target_points'], d_model = config['model_params']['d_model'], d_ff = config['model_params']['d_ff'], n_heads=config['model_params']['n_heads'], param_dict=param_dict)
        self.param_dict = param_dict
        self.dataloader = HistoryConcatLoaderSpace(
            file,
            param_dict,
            config
        )
        self.model = self.load_model()
        self.set_inv_scaler(self.dataloader.reverse_scaling)
        self.set_scaler(self.dataloader.apply_scaling)
        self.max_epochs = config['trainer_params']['max_epochs']
        print_network(self.model)

    def load_model(self):
        forecaster = self.spacetimeformer
        return forecaster

    def train_dataloader(self):
        return self.dataloader.load_train()

    def val_dataloader(self):
        return self.dataloader.load_valid()

    def test_dataloader(self):
        return self.dataloader.load_test()


def grid_search_Spacetimeformer(config, num_files):
    # set random seed
    torch.manual_seed(config['logging_params']['manual_seed'])
    torch.cuda.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    saved_folder = os.path.join(
        config['logging_params']['save_dir'], config['logging_params']['name'])
    flag = True
    while flag:
        if config['exp_params']['test_flag']:
            last_version = config['exp_params']['last_version'] - 1
        else:
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
                last_version = -1
            else:
                last_version = sorted([int(i.split('_')[1]) for i in os.listdir(
                    saved_folder) if i.startswith('version_')])[-1]
        log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
        if config['exp_params']['test_flag']:
            assert os.path.exists(log_dir)
            flag = False
        else:
            try:
                os.makedirs(log_dir)
                flag = False
            except:
                flag = True
    print(f'log_dir: {log_dir}')

    data_folder = config['exp_params']['data_folder']
    file_list = sorted([i for i in os.listdir(data_folder)
                       if 'zone' in i and i.endswith('.csv')])[:num_files]
    param_grid = {
        'sliding_window': config['exp_params']['sliding_window'],
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'normalization': config['exp_params']['normalization'],
        'layers': config['model_params']['layers'],
        'blocks': config['model_params']['blocks'],
        'dilation_channels': config['model_params']['dilation_channels'],
        'residual_channels': config['model_params']['residual_channels'],
        'skip_channels': config['model_params']['skip_channels'],
        'end_channels': config['model_params']['end_channels'],
        'kernel_size': config['model_params']['kernel_size'],
    }

    param_dict_list = list(ParameterGrid(param_grid))
    print("param_dict: ", param_dict_list)
    """
    getting validation results
    """
    for file in file_list:
        cur_log_dir = os.path.join(log_dir, file.split('.')[0])
        if not config['exp_params']['test_flag']:
            if not os.path.exists(cur_log_dir):
                os.makedirs(cur_log_dir)
            # print((Spacetimeformer_exp()))
            Pytorch_DNN_validation(os.path.join(
                data_folder, file), param_dict_list, cur_log_dir, config, Spacetimeformer_exp)
            """
            hyperparameters selection
            """
            summary = OrderedDict()
            for param_index, param_dict in enumerate(param_dict_list):
                param_dict = OrderedDict(param_dict)
                setting_name = 'param'
                for key, val in param_dict.items():
                    setting_name += f'_{key[0].capitalize()}{val}'

                model_list = [i for i in os.listdir(os.path.join(
                    cur_log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
                assert len(model_list) == 1
                perf = float(model_list[0][model_list[0].find(
                    'avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
                with open(os.path.join(cur_log_dir, setting_name, 'version_0', 'std.txt'), 'r') as f:
                    std_text = f.readlines()
                    std_list = [[int(i.split()[0]), list(
                        map(float, i.split()[1].split('_')))] for i in std_text]
                    std_dict = dict(
                        zip(list(zip(*std_list))[0], list(zip(*std_list))[1]))
                best_epoch = int(model_list[0][model_list[0].find(
                    'best-epoch=')+len('best-epoch='):model_list[0].find('-avg_val_metric')])
                std = std_dict[best_epoch]
                # perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('-std')])
                # std = float(model_list[0][model_list[0].find('-std=')+len('-std='):model_list[0].find('.ckpt')])
                summary['_'.join(map(str, list(param_dict.values())))] = [
                    perf, std]
            with open(os.path.join(cur_log_dir, 'val_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)

            selected_index = np.argmin(np.array(list(summary.values()))[:, 0])
            selected_params = list(summary.keys())[selected_index]
            param_dict = {
                'batch_size': int(selected_params.split('_')[0]),
                'blocks': int(selected_params.split('_')[1]),
                'dilation_channels': int(selected_params.split('_')[2]),
                'end_channels': int(selected_params.split('_')[3]),
                'kernel_size': int(selected_params.split('_')[4]),
                'layers': int(selected_params.split('_')[5]),
                'learning_rate': float(selected_params.split('_')[6]),
                'normalization': selected_params.split('_')[7],
                'residual_channels': int(selected_params.split('_')[8]),
                'skip_channels': int(selected_params.split('_')[9]),
                'sliding_window': int(selected_params.split('_')[10]),
                'std': np.array(list(summary.values()))[selected_index][-1],
            }
            # save param
            with open(os.path.join(cur_log_dir, 'param.json'), 'w') as f:
                json.dump(param_dict, f, indent=4)

        """
        prediction on testing
        """
        with open(os.path.join(cur_log_dir, 'param.json'), 'r') as f:
            param_dict = json.load(f)

        Pytorch_DNN_testing(os.path.join(data_folder, file),
                            param_dict, cur_log_dir, config, Spacetimeformer_exp)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # run evaluate
    evaluate_config = {
        'exp_params': {
            'prediction_path': log_dir,
            'prediction_interval': config['exp_params']['prediction_interval'],
            'data_folder': config['exp_params']['data_folder']
        }
    }
    run_evaluate_V3(config=evaluate_config, verbose=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--manual_seed', '-manual_seed',
                        type=int, help='random seed')
    parser.add_argument('--num_files', '-num_files', type=int,
                        default=3, help='number of files to predict')

    parser.add_argument('--sliding_window', '-sliding_window',
                        type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str,
                        help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio',
                        type=float, help='select hyperparameters on validation set')
    parser.add_argument('--external_features', '-external_features',
                        type=str, help='list of external feature name list')

    # model-specific features
    parser.add_argument('--batch_size', '-batch_size',
                        type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs',
                        type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate',
                        type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)  # , default='[0]')

    parser.add_argument('--layers', '-layers', type=str,
                        help='list of layers options')
    parser.add_argument('--blocks', '-blocks', type=str,
                        help='list of blocks options')
    parser.add_argument('--dilation_channels', '-dilation_channels',
                        type=str, help='list of dilation_channels options')
    parser.add_argument('--residual_channels', '-residual_channels',
                        type=str, help='list of residual_channels options')
    parser.add_argument('--skip_channels', '-skip_channels',
                        type=str, help='list of skip_channels options')
    parser.add_argument('--end_channels', '-end_channels',
                        type=str, help='list of end_channels options')
    parser.add_argument('--kernel_size', '-kernel_size',
                        type=str, help='list of kernel_size options')

    args = vars(parser.parse_args())
    with open('./../configs/spacetimeformer.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')
    print('gpu avail: ', torch.cuda.device_count())

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    # grid_search_Spacetimeformer(config, num_files=args['num_files'])
    grid_search_Spacetimeformer(config, num_files=1)
