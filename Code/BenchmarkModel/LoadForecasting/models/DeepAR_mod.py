### Import libraries
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

#os.chdir("../../..")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random

import pytorch_forecasting as ptf
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, RMSE, MAPE, MultivariateNormalDistributionLoss
from pytorch_forecasting.data import TorchNormalizer

from src.utils import config_compiler


def Deep_AR(param_combinations):
    ### Configure experiment
    try: df_results
    except NameError:
        df_results = None
    if df_results is None:
        df_results = pd.DataFrame(columns=['Parameters', 'LR', 'Val RMSE', 'Val MAE', 'Test RMSE', 'Val MAE'])

    for i in param_combinations:    
        temp_results = []

        year = i[0]
        max_enc = i[1]
        max_pred = i[2]
        hidden_state_size = i[3]
        num_layers = i[4]
        drop_out = i[5]
        batch_size = i[6]
        max_epochs = i[7]

        exp_name = '_'.join(str(v) for v in i)
        temp_results.append(exp_name)


        ### Set directories
        learn_rate_path = Path('results/learning_rate')
        performacne_path = Path('results/performance')
        data_path = Path('data')

        ### Input data
        data_file = 'CAISO_zone_1_' + year + '.csv'
        data_fp = data_path/data_file
        df = pd.read_csv(data_fp)
        df['holiday'] = df['holiday'].astype(str)

        ### Train, val, test
        train_cut_date = year + '-09-30 23:00:00'
        val_cut_date = year + '-11-30 23:00:00'
        train_cutoff_idx = df[df['date_time'] == train_cut_date]['time_idx'].values[0] # returns index of training cutoff
        val_cutoff_idx = df[df['date_time'] == val_cut_date]['time_idx'].values[0] # returns index of validation cutoff

        train_cutoff = train_cutoff_idx - max_pred

        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= train_cutoff], #seems more pandas/pythonic but leaving as is for now
            time_idx="time_idx",
            target="load_power",
            group_ids=['series'],
            time_varying_unknown_reals=["load_power"],
            time_varying_known_reals=["DNI_lag",
                                    "Dew Point_lag",
                                    "Solar Zenith Angle_lag",
                                    "Wind Speed_lag",
                                    "Relative Humidity_lag",
                                    "Temperature_lag", 'hour'],
            time_varying_known_categoricals= ['holiday'],
            max_encoder_length=max_enc,
            max_prediction_length=max_pred,
            target_normalizer=TorchNormalizer(method='identity', center=True, transformation=None, method_kwargs={}), # https://github.com/jdb78/pytorch-forecasting/issues/1220
            add_target_scales=True)

        validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.time_idx<=val_cutoff_idx], min_prediction_idx=train_cutoff_idx + 1)
        testing = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=val_cutoff_idx - 1)

        # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized")
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized")
        test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized")

        val_actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        test_actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])

        # Baseline estimator and absolute error on validation
        #val_baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
        #RMSE()(val_baseline_predictions, val_actuals)
        #test_baseline_predictions = Baseline().predict(test_dataloader, trainer_kwargs=dict(accelerator="cpu"))
        #RMSE()(test_baseline_predictions, test_actuals)


        ### Modeling 
        # Initiate model 
        pl.seed_everything(42)
        trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)
        net = DeepAR.from_dataset( # Look at create log
            training,
            learning_rate=3e-2,
            hidden_size=hidden_state_size,
            rnn_layers=num_layers,
            dropout = drop_out,
            loss=MultivariateNormalDistributionLoss(rank=30),
            optimizer="Adam")


        # Find optimal learning rate 
        # Error with this and mps https://github.com/pytorch/pytorch/issues/98074
        res = Tuner(trainer).lr_find(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            min_lr=1e-5,
            max_lr=1e0,
            early_stop_threshold=100)

        fig = res.plot(show=False, suggest=True)
        fig_name = exp_name + '.jpg'
        fig_fp = learn_rate_path/fig_name
        #fig.savefig(fig_fp)
        #print(f'Suggested learning rate {res.suggestion()}')
        net.hparams.learning_rate = res.suggestion()
        lr_round = round(res.suggestion(),4)
        temp_results.append(lr_round) 

        # Training
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        tensorboard = pl_loggers.TensorBoardLogger('./')
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches= batch_size,
            limit_val_batches=batch_size,
            enable_checkpointing=True,
            logger=tensorboard)

        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

        # Fit model - 
        #best_model_path = trainer.checkpoint_callback.best_model_path
        #best_model = DeepAR.load_from_checkpoint(best_model_path)

        # Predictions
        val_predictions = net.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"))
        val_rmse = RMSE()(val_predictions, val_actuals)
        val_mae = MAE()(val_predictions, val_actuals)
        temp_results.append(round(float(val_rmse.numpy()), 4))
        temp_results.append(round(float(val_mae.numpy()), 4))

        test_predictions = net.predict(test_dataloader, trainer_kwargs=dict(accelerator="cpu"))
        test_rmse = RMSE()(test_predictions, test_actuals)
        test_mae = MAE()(test_predictions, test_actuals)
        temp_results.append(round(float(test_rmse.numpy()), 4))
        temp_results.append(round(float(test_mae.numpy()), 4))

        # Add to results dataframe and save
        df_results.loc[len(df_results)] = temp_results
        result_name = exp_name + '.csv'
        result_fp = performacne_path/result_name
        #df_results.to_csv(result_fp, index=False)



_, configs = config_compiler('2020_best_hp.yaml')
Deep_AR(configs)

#_, configs = config_compiler('DeepAR.yaml')
#len(configs)
#configs_trim = random.choices(configs, k=50)
#Deep_AR(configs_trim)


#if __name__ == '__main__':
#    _, configs = config_compiler('DeepAR.yaml')
#    DeepAR(configs)
