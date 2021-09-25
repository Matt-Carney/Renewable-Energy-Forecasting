# Created by xunannancy at 2021/9/25
"""
three conventional rnn models: vanilla rnn, GRU, lstm
"""
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch
from FNN import FNN_exp
import os
import numpy as np
from sklearn.model_selection import ParameterGrid
from utils import Pytorch_DNN_validation, Pytorch_DNN_testing, merge_parameters, run_evaluate_V3
import json
from collections import OrderedDict
import argparse
import yaml

class RNNNet(nn.Module):
    def __init__(self,
                 sliding_window, external_features, history_column_names, target_val_column_names,
                 hidden_size, num_layers, dropout, direction, model_name, normalization):
        super().__init__()

        self.rnn_cell = model_name.upper()

        assert self.rnn_cell in ['RNN', 'LSTM', 'GRU']
        assert direction in ['uni', 'bi']
        self.sliding_window = sliding_window
        self.history_column_names, self.external_features = history_column_names, external_features
        self.direction = direction
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.normalization = normalization
        self.target_val_column_names = target_val_column_names

        self.embedding = nn.Sequential(
            nn.Linear(len(self.history_column_names) + len(self.external_features), hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

        if self.direction == 'uni':
            bidirectional = False
            self.cur_rnn_dim = self.hidden_size
        elif self.direction == 'bi':
            bidirectional = True
            self.cur_rnn_dim = int(self.hidden_size/2)

        self.rnn_layer = eval(f'nn.{self.rnn_cell}')(
            input_size=self.hidden_size,
            hidden_size=self.cur_rnn_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, len(self.target_val_column_names))
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshape = x.reshape([-1, self.sliding_window+1, len(self.external_features) + len(self.history_column_names)])
        embed = self.embedding(x_reshape)
        rnn_results = self.rnn_layer(embed)
        if self.rnn_cell in ['RNN', 'GRU']:
            hidden = rnn_results[1]
        elif self.rnn_cell == 'LSTM':
            hidden = rnn_results[1][0]
        """
        RNN: output, h_n
        GRU: output, h_n
        LSTM: output, (h_n, c_n)
        """
        pred = self.final_layer(hidden.reshape([(self.direction=='bi')+1, self.num_layers, batch_size, self.cur_rnn_dim])[:, -1, :].permute([1, 0, 2]).reshape([batch_size, self.hidden_size]))
        if self.normalization == 'minmax':
            pred = torch.sigmoid(pred)
        return pred

    def loss_function(self, batch):
        x, y, flag = batch
        """
        x: [#batch, #seqlen*#features]
        y/flag: [#batch, horizon]
        """
        pred = self.forward(x)
        if self.normalization == 'none':
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, torch.log(y)) * flag)
            pred = torch.exp(pred)
        else:
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, y) * flag)

        return loss, pred


class RNN_exp(FNN_exp):
    def __init__(self, file, param_dict, config):
        super().__init__(file, param_dict, config)

    def load_model(self):
        model = RNNNet(
            sliding_window=self.param_dict['sliding_window'],
            external_features=self.config['exp_params']['external_features'],
            history_column_names=self.dataloader.history_column_names,
            target_val_column_names=self.dataloader.target_val_column_names,
            hidden_size=self.param_dict['hidden_size'],
            num_layers=self.param_dict['num_layers'],
            dropout=self.param_dict['dropout'],
            direction=self.param_dict['direction'],
            model_name=self.config['model_params']['model_name'],
            normalization=self.param_dict['normalization']
        )
        return model

def grid_search_RNN(config, num_files):
    # set random seed
    torch.manual_seed(config['logging_params']['manual_seed'])
    torch.cuda.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    torch.backends.cudnn.enabled = False

    saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
    flag = True
    while flag:
        if config['exp_params']['test_flag']:
            last_version = config['exp_params']['last_version'] - 1
        else:
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
                last_version = -1
            else:
                last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder) if i.startswith('version_')])[-1]
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
    file_list = sorted([i for i in os.listdir(data_folder) if 'zone' in i and i.endswith('.csv')])[:num_files]
    param_grid = {
        'sliding_window': config['exp_params']['sliding_window'],
        'hidden_size': config['model_params']['hidden_size'],
        'num_layers': config['model_params']['num_layers'],
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'dropout': config['model_params']['dropout'],
        'direction': config['model_params']['direction'],
        'normalization': config['exp_params']['normalization'],
    }
    param_dict_list = list(ParameterGrid(param_grid))

    """
    getting validation results
    """
    for file in file_list:
        cur_log_dir = os.path.join(log_dir, file.split('.')[0])
        if not config['exp_params']['test_flag']:
            if not os.path.exists(cur_log_dir):
                os.makedirs(cur_log_dir)
            Pytorch_DNN_validation(os.path.join(data_folder, file), param_dict_list, cur_log_dir, config, RNN_exp)
            """
            hyperparameters selection
            """
            summary = OrderedDict()
            for param_index, param_dict in enumerate(param_dict_list):
                param_dict = OrderedDict(param_dict)
                setting_name = 'param'
                for key, val in param_dict.items():
                    setting_name += f'_{key[0].capitalize()}{val}'

                model_list = [i for i in os.listdir(os.path.join(cur_log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
                assert len(model_list) == 1
                perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
                with open(os.path.join(cur_log_dir, setting_name, 'version_0', 'std.txt'), 'r') as f:
                    std_text = f.readlines()
                    std_list = [[int(i.split()[0]), list(map(float, i.split()[1].split('_')))] for i in std_text]
                    std_dict = dict(zip(list(zip(*std_list))[0], list(zip(*std_list))[1]))
                best_epoch = int(model_list[0][model_list[0].find('best-epoch=')+len('best-epoch='):model_list[0].find('-avg_val_metric')])
                std = std_dict[best_epoch]
                # perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('-std')])
                # std = float(model_list[0][model_list[0].find('-std=')+len('-std='):model_list[0].find('.ckpt')])
                summary['_'.join(map(str, list(param_dict.values())))] = [perf, std]
            with open(os.path.join(cur_log_dir, 'val_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)

            selected_index = np.argmin(np.array(list(summary.values()))[:, 0])
            selected_params = list(summary.keys())[selected_index]
            param_dict = {
                'batch_size': int(selected_params.split('_')[0]),
                'direction': selected_params.split('_')[1],
                'dropout': float(selected_params.split('_')[2]),
                'hidden_size': int(selected_params.split('_')[3]),
                'learning_rate': float(selected_params.split('_')[4]),
                'normalization': selected_params.split('_')[5],
                'num_layers': int(selected_params.split('_')[6]),
                'sliding_window': int(selected_params.split('_')[7]),
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
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, RNN_exp)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # run evaluate
    evaluate_config = {
        'exp_params': {
            'prediction_path': log_dir,
            'prediction_interval': config['exp_params']['prediction_interval'],
        }
    }
    run_evaluate_V3(config=evaluate_config, verbose=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--manual_seed', '-manual_seed', type=int, help='random seed')
    parser.add_argument('--num_files', '-num_files', type=int, default=3, help='number of files to predict')

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--external_feature_flag', '-external_feature_flag', type=bool, help='whether to consider external features')
    parser.add_argument('--external_features', '-external_features', type=str, help='list of external feature name list')

    # model-specific features
    parser.add_argument('--hidden_size', '-hidden_size', type=str, help='list of hidden_size')
    parser.add_argument('--num_layers', '-num_layers', type=str, help='list of num_layers')
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')
    parser.add_argument('--model_name', '-model_name', type=str, help='one of [RNN, GRU, LSTM]')
    parser.add_argument('--direction', '-direction', type=str, help='one of [uni, bi]')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout rates')

    args = vars(parser.parse_args())
    with open('./../configs/RNN.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_RNN(config, num_files=args['num_files'])

