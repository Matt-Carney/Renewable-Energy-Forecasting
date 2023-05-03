import pandas as pd
import numpy as np 
from pathlib import Path
import yaml
import os
import matplotlib.pyplot as plt


def learning_curve_cleaner(year):
    deep_ar_fp = Path('results/learning_curves/deepar_subfolder')
    lc_fp = Path('results/learning_curves')
    train_lc_path = deep_ar_fp/f'train_loss_{year}.csv'
    val_lc_path = deep_ar_fp/f'./val_loss_{year}.csv'

    train_lc = pd.read_csv(train_lc_path).drop('Wall time', axis=1)
    val_lc = pd.read_csv(val_lc_path).drop('Wall time', axis=1)

    lc_df = train_lc.merge(val_lc, on='Step', suffixes=('_train', '_val'))

    lc_temp = lc_fp/f'deep_ar_lc_{year}.csv'
    lc_df.to_csv(lc_temp)

years = ['2018', '2019', '2020']
for y in years:
    learning_curve_cleaner(y)

# plt.plot(lc_df['Step'].to_numpy(), lc_df['Value_train'].to_numpy(), label='Training')
# plt.plot(lc_df['Step'].to_numpy(), lc_df['Value_val'].to_numpy(), label='Validation')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# #plt.ylim()
# plt.title('Train/Val Learning Curves')
# plt.legend()
# plt.show()