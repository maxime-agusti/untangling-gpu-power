import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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


# nvidia 535 drivers
driver_demo_535 = load_platform(filepath='data/250210-mig-muva-2xH100-burn300.csv',
                                drop_everything_before_measure='MIG_1g.12gb|1g.12gb|idle')
driver_demo_535['GPU'] = 'H100-NVL-94GB (400W) | driver 535.183.06'
# nvidia 570 drivers
driver_demo_570 = load_platform(filepath='data/250321-mig-muva-2xH100-burn300.csv',
                                drop_everything_before_measure='MIG_1g.12gb|1g.12gb|idle')
driver_demo_570['GPU'] = 'H100-NVL-94GB (400W) | driver 570.124.06'

driver_demo = pd.concat([driver_demo_535, driver_demo_570])
driver_demo['GI_normalized'] = driver_demo['GI'].apply(
    lambda x: normalize_label(x))
driver_demo['CI_normalized'] = driver_demo['CI'].apply(
    lambda x: normalize_label(x))
driver_demo = driver_demo.drop(
    driver_demo[driver_demo.CI.str.contains("c.")].index)

driver_demo = driver_demo.loc[driver_demo['compute'] > 0]

hue_order = driver_demo['CI'].unique()  # [:-1] # exlude 7g

g = sns.relplot(
    data=driver_demo,
    x='compute', y='SMI_power.draw',
    kind='line',
    hue='CI', hue_order=hue_order,
    style='CI', style_order=hue_order,
    col='GPU',
    markers=True, dashes=False,
    markersize=10,
    errorbar=None
)

sns.move_legend(g, "lower right", bbox_to_anchor=(.88, .15), frameon=True)
g._legend.set_title("CI", prop={'size': 14})
for text in g._legend.get_texts():
    text.set_fontsize(14)

g.set_titles(col_template="{col_name}", size=14)
g.set_ylabels("GPU power consumption (W)", clear_inner=False, fontsize=16)
g.set_xlabels("Compute slice allocated", clear_inner=False, fontsize=16)
plt.gcf().savefig('figures/MIG-driver.pdf', bbox_inches='tight')
