import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.color_palette("Set2")
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

plt.tight_layout()

dataset = dataset.drop(dataset.loc[dataset['timestamp'] > 280].index)
dataset = dataset.drop(dataset.loc[dataset['timestamp'] < 30].index)

ipmi_keys = [
    key for key in dataset_keys if 'IPMI' in key and 'GPU' in key and not key.endswith('Zone-')]
domain = ['GPU0', 'GPU1', 'GPU2', 'GPU3']
domain_ipmi_keys = domain + ipmi_keys
timestamp_ipmi_keys = ['timestamp'] + domain_ipmi_keys

corr = dataset[domain_ipmi_keys].corr(method='pearson')
corr_filtered = corr[:len(domain)][ipmi_keys].T.dropna()

fig, ax = plt.subplots(figsize=(5, 7))  # Set the figure size
plt.tight_layout()

sns.heatmap(corr_filtered, ax=ax, annot=True, cmap=sns.color_palette(
    palette='RdGy_r', as_cmap=True), vmin=-1, vmax=1)

plt.gcf().savefig('figures/PT-pearson-corr-ipmi.pdf', bbox_inches='tight')
