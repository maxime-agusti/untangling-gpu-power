import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('data/250206-passthrough-chuc-4xA100-burn300.csv')
df = pd.read_csv('data/250226-passthrough-chuc4-4xA100-burn300-4states.csv')
df.reset_index(inplace=True)

drop_everything_before_measure = 'idle'

timestamp_decrease = df.timestamp.diff() < 0
restart = timestamp_decrease[timestamp_decrease].index.values
range_start = df.loc[df['measure'] ==
                     drop_everything_before_measure].iloc[0].values[0].astype(int)
frames = []
for range_end in restart:
    if range_start > 0:
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

dataset = pd.concat(frames)
dataset.set_index(['timestamp', 'domain', 'CONST_context'], inplace=True)
dataset_keys = list(dataset.keys())
if 'SMI_PSTATE' in dataset_keys:
    dataset_keys.remove('SMI_PSTATE')
if 'SMI_pstate' in dataset_keys:
    dataset_keys.remove('SMI_pstate')
dataset = dataset.astype({key: 'float' for key in dataset_keys})
dataset.reset_index(inplace=True)

dataset = dataset.drop(dataset.loc[dataset['timestamp'] > 280].index)
dataset = dataset.drop(dataset.loc[dataset['timestamp'] < 30].index)

ipmi_keys = [
    key for key in dataset_keys if 'IPMI' in key and 'GPU' in key and not key.endswith('Zone-')]
domain = ['GPU0', 'GPU1', 'GPU2', 'GPU3']
domain_ipmi_keys = domain + ipmi_keys
timestamp_ipmi_keys = ['timestamp'] + domain_ipmi_keys

corr = dataset[domain_ipmi_keys].corr(method='pearson')
corr_filtered = corr[:len(domain)][ipmi_keys].T.dropna()

ipmi_correlation = {}
for gpu in domain:
    ipmi_correlation[gpu] = corr_filtered[gpu].idxmax()
    print('For', gpu, 'selecting', ipmi_correlation[gpu])


def attach_sensor_value(row):
    if row['domain'] not in ipmi_correlation.keys():
        return None
    return dataset[(dataset["timestamp"] == row["timestamp"]) & (dataset["CONST_context"] == row["CONST_context"]) & (dataset["domain"] == "GPU-X")][ipmi_correlation[row["domain"]]].values[0]


dataset['sensor'] = dataset.apply(lambda row: attach_sensor_value(row), axis=1)
dataset_domains = dataset.loc[dataset['domain'].isin(ipmi_correlation.keys())]

dataset_domains = dataset.drop(
    dataset.loc[dataset['domain'].isin(['GPU-X', 'global'])].index)
dataset_domains['compute'] = dataset_domains.apply(
    lambda x: x['CONST_context'].split('|')[int(x['domain'].replace("GPU", ""))], axis=1)

corr_neighbour = dataset[ipmi_correlation.values()].dropna().corr(method='spearman').rename(dict(
    (v, k) for k, v in ipmi_correlation.items())).rename(dict((v, k) for k, v in ipmi_correlation.items()), axis=1)

neighbour = {}
for gpu in domain:
    neighbour_gpu = corr_neighbour[corr_neighbour.index != gpu][gpu].idxmax()
    neighbour[gpu] = ipmi_correlation[neighbour_gpu]
    print('For', gpu, 'neighbour might be', neighbour_gpu)


def attach_neighbour_value(row):
    if row['domain'] not in neighbour.keys():
        return None
    return dataset[(dataset["timestamp"] == row["timestamp"]) & (dataset["CONST_context"] == row["CONST_context"]) & (dataset["domain"] == "GPU-X")][neighbour[row["domain"]]].values[0]


dataset['neighbour'] = dataset.apply(
    lambda row: attach_neighbour_value(row), axis=1)
dataset_domains = dataset.loc[dataset['domain'].isin(neighbour.keys())]
dataset_domains['neighbour_delta'] = dataset_domains.apply(
    lambda row: row['neighbour'] - dataset[neighbour[row["domain"]]].min(), axis=1)
dataset_domains['compute'] = dataset_domains.apply(
    lambda x: x['CONST_context'].split('|')[int(x['domain'].replace("GPU", ""))], axis=1)

neighbour_delta_dict = {}

# Neighbour delta is computed between its current and min
# It is far better than computing the delta based on sensor.

for gpu in dataset_domains['domain'].unique():
    myloc = dataset_domains.loc[dataset_domains['domain'].eq(
        gpu) & dataset_domains['compute'].eq('2')]
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


dataset_domains['corrected'] = dataset_domains.apply(
    lambda row: correct_row(row), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plt.tight_layout()

g = sns.kdeplot(dataset_domains, x="sensor", hue="compute",
                fill=True, ax=axes[0], hue_order=['0', '2', '4', '7'])
axes[0].set(xlabel='Closest IPMI sensor value (°C)', ylabel='Density')
axes[0].legend(handles=axes[0].get_legend().legend_handles, labels=[
               '0%', '~30%', '~60%', '100%'], title="GPU usage")

g = sns.kdeplot(dataset_domains, x="corrected",
                hue="compute", fill=True, ax=axes[1])
axes[1].set(xlabel='Closest IPMI sensor value corrected (°C)', ylabel='Density')
axes[1].legend(handles=axes[1].get_legend().legend_handles, labels=[
               '0%', '~30%', '~60%', '100%'], title="GPU usage")

plt.gcf().savefig('figures/PT-density-4A100-4states-corrected.pdf', bbox_inches='tight')
