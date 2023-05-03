import pandas as pd
import numpy as np 
from pathlib import Path
import yaml
import os
import matplotlib.pyplot as plt


lc_fp = Path('results/learning_curves')
fig_fp = lc_fp/'figures'




def tb_cleaner(df):
    df['Epoch'] = df['Unnamed: 0'] + 1
    df = df.rename(columns= {'Value_train' :'train_loss',
                            'Value_val': 'val_loss'}) 
    return df


def plot(title, df):
    plt.figure(figsize=(6,4))
    if 'Temporal' in title:
        plt.xticks(np.arange(1, df['Epoch'].max()+1, 2.0))
    plt.plot(df['Epoch'], df['train_loss'], label='Training')
    plt.plot(df['Epoch'], df['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim()
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig(fig_fp/f'{title}.jpg')
    plt.close()

df_tft_2018 = pd.read_csv(lc_fp/'learning_curves_2018.csv')
df_tft_2018 = tb_cleaner(df_tft_2018)
plot('Temporal Fusion Transformer Learning Curve - 2018', df_tft_2018)
df_tft_2019 = pd.read_csv(lc_fp/'learning_curves_2019.csv')
df_tft_2019 = tb_cleaner(df_tft_2019)
plot('Temporal Fusion Transformer Learning Curve - 2019', df_tft_2019)
df_tft_2020 = pd.read_csv(lc_fp/'learning_curves_2020.csv')
df_tft_2020 = tb_cleaner(df_tft_2020)
plot('Temporal Fusion Transformer Learning Curve - 2020', df_tft_2020)


# df_deep_2018 = pd.read_csv(lc_fp/'deep_ar_lc_2018.csv')
# df_deep_2018 = tb_cleaner(df_deep_2018)
# plot('DeepAR Learning Curve - 2018', df_deep_2018)
# df_deep_2019 = pd.read_csv(lc_fp/'deep_ar_lc_2019.csv')
# df_deep_2019 = tb_cleaner(df_deep_2019)
# plot('DeepAR Learning Curve - 2019', df_deep_2019)
# df_deep_2020 = pd.read_csv(lc_fp/'deep_ar_lc_2020.csv')
# df_deep_2020 = tb_cleaner(df_deep_2020)
# plot('DeepAR Learning Curve - 2020', df_deep_2020)

# df_st_2018 = pd.read_csv(lc_fp/'Pei_loss_metrics_2018.csv')
# df_st_2019 = pd.read_csv(lc_fp/'Pei_loss_metrics_2019.csv')
# df_st_2020 = pd.read_csv(lc_fp/'Pei_loss_metrics_2020.csv')
# df_st_2018['Epoch'] = df_st_2018['Epoch']+1
# df_st_2019['Epoch'] = df_st_2019['Epoch']+1
# df_st_2020['Epoch'] = df_st_2020['Epoch']+1
# plot('Spacetimeformer Learning Curve - 2018', df_st_2018)
# plot('Spacetimeformer Learning Curve - 2019', df_st_2019)
# plot('Spacetimeformer Learning Curve - 2020', df_st_2020)

