import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

data_dir = 'data'
filenames = os.listdir(data_dir)


def compute_gpu_data(gpu_model: str, filtered=False):
    gpu_filenames = [
        filename
        for filename in filenames
        if f'ovh-1x{gpu_model}' in filename
    ]

    def read_csv(filename: str, filtered=False):
        df = pd.read_csv(f'{data_dir}/{filename}')
        df['source'] = filename

        if filtered:
            # Filter the lenght of experiments to remove outliers
            return df.loc[df['timestamp'].between(30, 300)]
        else:
            return df

    gpu_data = pd.concat(
        [
            read_csv(filename, filtered=filtered)
            for filename in gpu_filenames
        ],
        ignore_index=True,
    )

    timestamp_diff = gpu_data['timestamp'].diff()
    timestamp_reset = gpu_data[timestamp_diff < 0]

    gpu_data.loc[
        timestamp_diff < 0,
        'stage',
    ] = range(1, len(timestamp_reset) + 1)

    gpu_data['stage'] = gpu_data['stage'].ffill()
    gpu_data['stage'] = gpu_data['stage'].fillna(0)

    timestamp_1 = gpu_data['timestamp'].shift(1)

    idx = timestamp_1.loc[gpu_data.drop_duplicates(subset='stage').index] + 1
    idx = idx.fillna(0)
    idx = idx.cumsum()
    idx = idx.reindex(gpu_data.index)
    idx = idx.ffill()
    idx = idx + gpu_data['timestamp']

    gpu_data['timedelta'] = pd.to_timedelta(idx, unit='s')

    gpu_data = gpu_data[gpu_data['metric'].isin([
        'SMI_temperature.gpu',
        'SMI_power.draw',
        'CONST_context',
    ])].copy()
    # gpu_data['measure'] = gpu_data['measure'].astype(float)

    gpu_data_pivot = gpu_data.pivot_table(
        index=['timedelta', 'source'],
        columns='metric',
        values='measure',
        aggfunc='first',
    )

    gpu_data_pivot['gpu_model'] = gpu_model

    return gpu_data_pivot.reset_index()


def load(filtered=False):
    all_gpu_data = pd.concat(
        [compute_gpu_data(gpu_model, filtered=filtered)
         for gpu_model in ['H100', 'A100']],
        ignore_index=True,
    )

    all_gpu_data['SMI_power.draw'] = all_gpu_data['SMI_power.draw'].astype(
        float)
    all_gpu_data['SMI_temperature.gpu'] = all_gpu_data['SMI_temperature.gpu'].astype(
        float)

    max_split = 7

    parsed_ctx = all_gpu_data['CONST_context'].str.split(
        '|', expand=True).rename(columns={0: 'GI', 1: 'CI', 2: 'noise'})
    parsed_ctx = parsed_ctx.dropna()

    compute = parsed_ctx.apply(
        lambda x: 0 if x['noise'] == 'idle' else (
            int(x['noise']) + int(re.match(r'^\d+', x['CI']).group())),
        axis=1,
    )

    all_gpu_data['target_utilization'] = (
        (compute / max_split) * 100).round().astype(int)
    return all_gpu_data.dropna(subset='target_utilization')


all_gpu_data = load()
all_gpu_data_filtered = load(filtered=True)

pwr_window = 50

fig1_data = all_gpu_data.copy()
fig1_data = fig1_data.sort_values(by='gpu_model', ascending=False)
fig1_data['gpu_power_cut'] = (
    fig1_data['SMI_power.draw'] / pwr_window
).round() * pwr_window
fig1_data['gpu_model_explicit'] = fig1_data['gpu_model'].apply(
    lambda x: 'H100-PCIE-80GB (350W)' if x == 'H100' else 'A100-PCIE-80GB (300W)')

power_cut_values = fig1_data['gpu_power_cut'].sort_values().unique()
utilization_values = fig1_data['target_utilization'].sort_values().unique()

fig1_data_gby_gpu = fig1_data.groupby('gpu_model')

fig2_data = all_gpu_data_filtered.copy()
fig2_data = fig2_data.sort_values(by='gpu_model', ascending=False)
fig2_data['gpu_power_cut'] = (
    fig2_data['SMI_power.draw'] / pwr_window
).round() * pwr_window
fig2_data['gpu_model_explicit'] = fig2_data['gpu_model'].apply(
    lambda x: 'H100-PCIE-80GB (350W)' if x == 'H100' else 'A100-PCIE-80GB (300W)')

power_cut_values = fig2_data['gpu_power_cut'].sort_values().unique()
utilization_values = fig2_data['target_utilization'].sort_values().unique()

fig2_data_gby_gpu = fig2_data.groupby('gpu_model')

a100_line_color = sns.color_palette("pastel")[1]

fig, axs = plt.subplots(1, 2)

fig1_data['gpu_power_cut_n'] = fig1_data['gpu_power_cut'].round().astype(int)
sns.boxenplot(
    data=fig1_data,
    x='gpu_power_cut_n',
    y='SMI_temperature.gpu',
    hue='gpu_model_explicit',
    showfliers=False,
    ax=axs[0],
)

axs[0].set_ylabel('GPU temperature (°C)')
axs[0].set_xlabel('GPU power consumption (W)')
axs[0].legend().set_loc('upper left')

fig2_data['target_utilization_n'] = fig2_data['target_utilization'].round().astype(int)
sns.boxenplot(
    data=fig2_data,
    x='target_utilization_n',
    y='SMI_temperature.gpu',
    hue='gpu_model_explicit',
    showfliers=False,
    ax=axs[1],
)


axs[1].set_ylabel(None)
axs[1].yaxis.set_ticklabels([])
axs[1].set_xlabel('GPU utilization (%)')
axs[1].legend().remove()

fig.set_figwidth(10)

fig.tight_layout()
fig.savefig('figures/WC-GPU-temp-pwr-util.pdf')
