import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

sns.color_palette("Set2")
sns.set_theme(style="darkgrid")


def load_platform(filepath, drop_everything_before_measure):

    df = pd.read_csv(filepath)
    df.reset_index(inplace=True)

    timestamp_decrease = df.timestamp.diff() < 0
    restart = timestamp_decrease[timestamp_decrease].index.values
    range_start = df.loc[df['measure'] ==
                         drop_everything_before_measure].iloc[0].values[0].astype(int)
    frames = []
    for range_end in restart:
        if range_start > 0:
            frames.append(df[range_start:range_end].pivot(
                index=['timestamp', 'domain'], columns='metric', values='measure').reset_index())
        range_start = range_end
    frames.append(df[range_start:].pivot(
        index=['timestamp', 'domain'], columns='metric', values='measure').reset_index())

    dataset = pd.concat(frames)
    dataset.set_index(['timestamp', 'domain', 'CONST_context'], inplace=True)
    keys_to_drop = [x for x in list(
        dataset.keys()) if 'IPMI' in x and '_Temp_' not in x]
    dataset.drop(keys_to_drop, axis=1, inplace=True)
    dataset.replace('No', None, inplace=True)
    dataset_keys = list(dataset.keys())
    if 'SMI_PSTATE' in dataset_keys:
        dataset_keys.remove('SMI_PSTATE')
    if 'SMI_pstate' in dataset_keys:
        dataset_keys.remove('SMI_pstate')
    dataset = dataset.astype({key: 'float' for key in dataset_keys})
    dataset.reset_index(inplace=True)

    dataset_domain = dataset.drop(
        dataset.loc[dataset['domain'].isin(['GPU-X', 'global'])].index)
    dataset_domain = dataset_domain.join(dataset_domain['CONST_context'].str.split(
        '|', expand=True).rename(columns={0: 'GI', 1: 'CI', 2: 'noise'}))
    dataset_domain['wrk'] = dataset_domain['noise'].apply(
        lambda x: 0 if x == 'idle' else int(x) + 1)
    dataset_domain['compute'] = dataset_domain.apply(lambda x: 0 if x['noise'] == 'idle' else (
        int(x['noise']) + int(re.match(r"^\d+", x['CI']).group())),  axis=1)
    dataset_domain = dataset_domain.drop(
        dataset_domain.loc[dataset_domain['timestamp'] > 180].index)
    dataset_domain = dataset_domain.drop(
        dataset_domain.loc[dataset_domain['timestamp'] < 30].index)
    # dataset_domain_pure = dataset_domain.drop(dataset_domain[dataset_domain.CI.str.contains("c.")].index)

    return dataset_domain


def separate_idle_load(dataset_domain, compute_total, load_gi):
    idle = dataset_domain.loc[dataset_domain['noise'] == 'idle']
    idle["SMI_power.draw-norm"] = idle["SMI_power.draw"] / compute_total
    idle['label'] = 'idle'
    idle_value = np.median(idle["SMI_power.draw"])

    load = dataset_domain.loc[(dataset_domain['noise'] == '0') & (
        dataset_domain['GI'] == load_gi)]
    load["SMI_power.draw-norm"] = load["SMI_power.draw"] - idle_value
    load['label'] = 'load'
    load_value = np.median(load["SMI_power.draw"])

    return idle_value, load_value, pd.concat([idle, load])


dataset_domain = load_platform(filepath='data/250321-mig-ovh-1xH100-burn300.csv',
                               drop_everything_before_measure='MIG_1g.10gb|1g.10gb|idle')

# A100
a100_grouille = load_platform(filepath='data/250320-mig-grouille-4xA100-burn300.csv',
                              drop_everything_before_measure='MIG_1g.5gb|1g.5gb|idle')
a100_grouille['GPU'] = 'A100-PCIE-40GB (250W)'
_, _, a100_grouille_f = separate_idle_load(
    dataset_domain=a100_grouille, compute_total=7, load_gi='MIG_1g.5gb')

a100_ovh = load_platform(filepath='data/250320-mig-ovh-1xA100-burn300.csv',
                         drop_everything_before_measure='MIG_1g.10gb|1g.10gb|idle')
a100_ovh['GPU'] = 'A100-PCIE-80GB (300W)'
_, _, a100_ovh_f = separate_idle_load(
    dataset_domain=a100_ovh, compute_total=7, load_gi='MIG_1g.10gb')

a100_chuc = load_platform(filepath='data/250320-mig-chuc-4xA100-burn300.csv',
                          drop_everything_before_measure='MIG_1g.5gb|1g.5gb|idle')
a100_chuc['GPU'] = 'A100-SXM4-40GB (400W)'
_, _, a100_chuc_f = separate_idle_load(
    dataset_domain=a100_chuc, compute_total=7, load_gi='MIG_1g.5gb')

# H100
h100_ovh = load_platform(filepath='data/250321-mig-ovh-1xH100-burn300.csv',
                         drop_everything_before_measure='MIG_1g.10gb|1g.10gb|idle')
h100_ovh['GPU'] = 'H100-PCIE-80GB (350W)'
_, _, h100_ovh_f = separate_idle_load(
    dataset_domain=h100_ovh, compute_total=7, load_gi='MIG_1g.10gb')

h100_muva = load_platform(filepath='data/250321-mig-muva-2xH100-burn300.csv',
                          drop_everything_before_measure='MIG_1g.12gb|1g.12gb|idle')
h100_muva['GPU'] = 'H100-NVL-94GB (400W)'
_, _, h100_muva_f = separate_idle_load(
    dataset_domain=h100_muva, compute_total=7, load_gi='MIG_1g.12gb')

load_gpus = pd.concat(
    [a100_grouille, a100_ovh, a100_chuc, h100_ovh, h100_muva])
slice_gpus = pd.concat([a100_grouille_f, a100_ovh_f, a100_chuc_f,
                       h100_ovh_f, h100_muva_f])  # , gh200_hydra_f])


def normalize_label(label):
    labels = label.split('.')
    if len(labels) == 2:
        slicex = ''
        compute, mem = labels
    else:
        slicex, compute, mem = labels
        slicex = '|' + slicex
    result = compute.lower().replace('mig_', '') + slicex
    if result == '1g' and mem in ['10gb', '24gb']:
        return result + '.x2GB'
    return result


load_gpus['GI_normalized'] = load_gpus['GI'].apply(
    lambda x: normalize_label(x))
load_gpus['CI_normalized'] = load_gpus['CI'].apply(
    lambda x: normalize_label(x))

load_gpus_pure = load_gpus.drop(
    load_gpus[load_gpus.CI.str.contains("c.")].index)

hue_order = load_gpus_pure['CI_normalized'].unique()[:-1]  # exlude 7g

g = sns.relplot(data=load_gpus_pure, x='compute', y='SMI_power.draw', kind='line', hue='CI_normalized',
                hue_order=hue_order, style='CI_normalized', style_order=hue_order, col='GPU', markers=True, dashes=False)
sns.move_legend(g, "lower right", bbox_to_anchor=(.94, .15), frameon=True)
g._legend.set(title='CI normalized')


g.set_ylabels("GPU power consumption (W)", clear_inner=False)
g.set_xlabels("Compute slice allocated", clear_inner=False)

plt.gcf().savefig('figures/MIG-GI-power-all.pdf', bbox_inches='tight')
