import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv(
    'data/250209-passthrough-sirius-8xA100-burn300-2states.csv')
df.reset_index(inplace=True)

# df = df.drop(df.loc[df['measure'] == 'No' ].index)
drop_everything_before_measure = 'idle'

timestamp_decrease = df.timestamp.diff() < 0
restart = timestamp_decrease[timestamp_decrease].index.values
range_start = df.loc[df['measure'] ==
                     drop_everything_before_measure].iloc[0].values[0].astype(int)
frames = []
for range_end in restart:
    if range_start > 0:
        range_start = df[range_start:range_end].loc[(df[range_start:range_end]['timestamp'] == 0) & (
            df[range_start:range_end]['domain'] == 'global') & (df[range_start:range_end]['metric'] == 'CONST_context')].iloc[-1].values[0].astype(int)
        focus = df[range_start:range_end].pivot(
            index=['timestamp', 'domain'], columns='metric', values='measure')
        focus.reset_index(inplace=True)
        pivot = focus.pivot(index='timestamp',
                            columns='domain', values='SMI_power.draw')
        frames.append(pd.concat([pivot, focus], axis=1))
    range_start = range_end
focus = df[range_start:].pivot(
    index=['timestamp', 'domain'], columns='metric', values='measure')
focus.reset_index(inplace=True)
pivot = focus.pivot(index='timestamp', columns='domain',
                    values='SMI_power.draw')
frames.append(pd.concat([pivot, focus], axis=1))

dataset2 = pd.concat(frames)
dataset2.set_index(['timestamp', 'domain', 'CONST_context'], inplace=True)
dataset_keys = list(dataset2.keys())
if 'SMI_PSTATE' in dataset_keys:
    dataset_keys.remove('SMI_PSTATE')
if 'SMI_pstate' in dataset_keys:
    dataset_keys.remove('SMI_pstate')
for key in dataset_keys:
    dataset2[key] = dataset2[key].replace('No', None)
    dataset2 = dataset2.astype({key: 'float'})
dataset2.reset_index(inplace=True)
dataset2 = dataset2[dataset2['CONST_context'].notna()]

dataset2 = dataset2.drop(dataset2.loc[dataset2['timestamp'] > 280].index)
dataset2 = dataset2.drop(dataset2.loc[dataset2['timestamp'] < 30].index)


ipmi_keys = [key for key in dataset_keys if 'IPMI' in key]
domain = ['GPU0', 'GPU1', 'GPU2', 'GPU3', 'GPU4', 'GPU5', 'GPU6', 'GPU7']
domain_ipmi_keys = domain + ipmi_keys
timestamp_ipmi_keys = ['timestamp'] + domain_ipmi_keys

corr2 = dataset2[domain_ipmi_keys].corr(method='pearson')
corr_filtered2 = corr2[:len(domain)][ipmi_keys].T.dropna()

ipmi_correlation = {}
for gpu in domain:
    ipmi_correlation[gpu] = corr_filtered2[gpu].idxmax()
    print('For', gpu, 'selecting', ipmi_correlation[gpu])


def attach_sensor_value(row):
    if row['domain'] not in ipmi_correlation.keys():
        return None
    return dataset2[(dataset2["timestamp"] == row["timestamp"]) & (dataset2["CONST_context"] == row["CONST_context"]) & (dataset2["domain"] == "GPU-X")][ipmi_correlation[row["domain"]]].values[0]


dataset2['sensor'] = dataset2.apply(
    lambda row: attach_sensor_value(row), axis=1)
dataset_domains2 = dataset2.loc[dataset2['domain'].isin(
    ipmi_correlation.keys())]

dataset_domains2 = dataset2.drop(
    dataset2.loc[dataset2['domain'].isin(['GPU-X', 'global'])].index)
dataset_domains2['compute'] = dataset_domains2.apply(
    lambda x: x['CONST_context'].split('|')[int(x['domain'].replace("GPU", ""))], axis=1)

corr_neighbour = dataset2[ipmi_correlation.values()].dropna().corr(method='spearman').rename(dict(
    (v, k) for k, v in ipmi_correlation.items())).rename(dict((v, k) for k, v in ipmi_correlation.items()), axis=1)

neighbour = {}
for gpu in domain:
    neighbour_gpu = corr_neighbour[corr_neighbour.index != gpu][gpu].idxmax()
    neighbour[gpu] = ipmi_correlation[neighbour_gpu]
    print('For', gpu, 'neighbour might be', neighbour_gpu,
          corr_neighbour[corr_neighbour.index != gpu][gpu].max())


def attach_neighbour_value(row):
    if row['domain'] not in neighbour.keys():
        return None
    return dataset2[(dataset2["timestamp"] == row["timestamp"]) & (dataset2["CONST_context"] == row["CONST_context"]) & (dataset2["domain"] == "GPU-X")][neighbour[row["domain"]]].values[0]


dataset2['neighbour'] = dataset2.apply(
    lambda row: attach_neighbour_value(row), axis=1)
dataset_domains2 = dataset2.loc[dataset2['domain'].isin(neighbour.keys())]
dataset_domains2['neighbour_delta'] = dataset_domains2.apply(
    lambda row: row['neighbour'] - dataset2[neighbour[row["domain"]]].min(), axis=1)
dataset_domains2['compute'] = dataset_domains2.apply(
    lambda x: x['CONST_context'].split('|')[int(x['domain'].replace("GPU", ""))], axis=1)

neighbour_delta_dict = {}
dataset_domains2['neighbour_delta'] = dataset_domains2['neighbour'] - \
    dataset_domains2['sensor']
for gpu in dataset_domains2['domain'].unique():
    myloc = dataset_domains2.loc[dataset_domains2['domain'].eq(
        gpu) & dataset_domains2['compute'].eq('0')]
    a, b = np.polyfit(myloc['neighbour_delta'], myloc['sensor'], 1)
    if a > 0.1:  # we are only interestend on positive, non constant, relations
        print(gpu, a, b)
        neighbour_delta_dict[gpu] = a, b


def correct_row(row):
    if row['domain'] not in neighbour_delta_dict:
        return row['sensor']
    a, b = neighbour_delta_dict[row['domain']]
    delta = a*row['neighbour_delta']
    if delta < 0:
        delta = 0
    return row['sensor'] - delta


dataset_domains2['corrected'] = dataset_domains2.apply(
    lambda row: correct_row(row), axis=1)


fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plt.tight_layout()

g = sns.kdeplot(dataset_domains2, x="sensor", hue="compute", fill=True, palette=[
                sns.color_palette()[3], sns.color_palette()[0]], ax=axes[0])
axes[0].set(xlabel='Closest IPMI sensor value (°C)', ylabel='Density')
axes[0].legend(['0%', '100%'], title="GPU usage")

g = sns.kdeplot(dataset_domains2, x="corrected", hue="compute", fill=True, palette=[
                sns.color_palette()[3], sns.color_palette()[0]], ax=axes[1])
axes[1].set(xlabel='Closest IPMI sensor value corrected (°C)', ylabel='Density')
axes[1].legend(['0%', '100%'], title="GPU usage")

plt.gcf().savefig('figures/PT-density-8A100-2states-corrected.pdf', bbox_inches='tight')
