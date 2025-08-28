import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.color_palette("Set2")
sns.set_theme(style="darkgrid")


def load_platform_df(file: str):

    df = pd.read_csv(file)
    df.reset_index(inplace=True)

    drop_everything_before_measure = 'idle'

    timestamp_decrease = df.timestamp.diff() < 0
    restart = timestamp_decrease[timestamp_decrease].index.values
    # df.loc[df['measure'] == drop_everything_before_measure].iloc[0].values[0].astype(int)
    range_start = 0
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
    dataset_keys = list(dataset.keys())
    if 'SMI_PSTATE' in dataset_keys:
        dataset_keys.remove('SMI_PSTATE')
    if 'SMI_pstate' in dataset_keys:
        dataset_keys.remove('SMI_pstate')
    dataset = dataset.astype({key: 'float' for key in dataset_keys})
    dataset.reset_index(inplace=True)

    dataset_domains = dataset.drop(
        dataset.loc[dataset['domain'].isin(['GPU-X', 'global'])].index)
    dataset_domains['bench'] = dataset_domains['CONST_context'].apply(
        lambda x: x.split('|')[0])
    dataset_domains['instances'] = dataset_domains['CONST_context'].apply(
        lambda x: x.split('|')[1])
    return dataset_domains


def load_blender(file):
    if not os.path.isfile(file):
        return None

    blender_header = ['timestamp', 'context', 'gpu-model', 'device_peak_memory',
                      'number_of_samples', 'time_for_samples', 'samples_per_minute']
    blender_df = pd.read_csv(file, names=blender_header, header=None)

    blender_df['bench'] = blender_df['context'].apply(
        lambda x: x.split('|')[0])
    blender_df['instances'] = blender_df['context'].apply(
        lambda x: int(x.split('|')[1]))
    blender_df['domain'] = 'GPU' + \
        blender_df['context'].apply(lambda x: x.split('|')[2])
    return blender_df


def load_hpcg(file):
    if not os.path.isfile(file):
        return None

    hpcg_header = ['timestamp', 'context', 'metric', 'measure']
    hpcg_df = pd.read_csv(file, names=hpcg_header, header=None).reset_index()

    hpcg_df = hpcg_df.pivot_table(
        index=['timestamp', 'context'], columns='metric', values='measure', aggfunc='mean').reset_index()

    hpcg_df['bench'] = hpcg_df['context'].apply(lambda x: x.split('|')[0])
    hpcg_df['instances'] = hpcg_df['context'].apply(
        lambda x: int(x.split('|')[1]))
    hpcg_df['domain'] = 'GPU' + \
        hpcg_df['context'].apply(lambda x: x.split('|')[2])
    return hpcg_df


def load_llama(file):
    if not os.path.isfile(file):
        return None

    llama_header = ['timestamp', 'context', 'measure']
    llama_df = pd.read_csv(file, names=llama_header, header=None).reset_index()

    llama_df['bench'] = llama_df['context'].apply(lambda x: x.split('|')[0])
    llama_df['instances'] = llama_df['context'].apply(
        lambda x: int(x.split('|')[1]))
    llama_df['domain'] = 'GPU' + \
        llama_df['context'].apply(lambda x: x.split('|')[2])
    return llama_df


def load_yolo(file):
    if not os.path.isfile(file):
        return None

    yolo_header = ['timestamp', 'context', 'measure']
    yolo_df = pd.read_csv(file, names=yolo_header, header=None).reset_index()

    yolo_df['bench'] = yolo_df['context'].apply(lambda x: x.split('|')[0])
    yolo_df['instances'] = yolo_df['context'].apply(
        lambda x: int(x.split('|')[1]))
    yolo_df['domain'] = 'GPU' + \
        yolo_df['context'].apply(lambda x: x.split('|')[2])
    return yolo_df


def correlate_perf_power(metric_df, platform_df, bench: str, metric_perf: str):
    res = {'domain': [], 'perf': [], 'perf%': [],
           'power': [], 'power%': [], 'instances': []}
    for domain in metric_df['domain'].unique():
        for instances in metric_df.loc[metric_df['domain'] == domain]['instances'].unique():
            perf = metric_df.loc[(metric_df['domain'] == domain) & (
                metric_df['instances'] == instances)][metric_perf].median()
            power = platform_df.loc[(platform_df['domain'] == domain) & (platform_df['bench'] == bench) & (
                platform_df['instances'] == str(instances))]['SMI_power.draw'].median() / instances
            res['domain'].append(domain)
            res['perf'].append(perf)
            perf_baseline = metric_df.loc[(metric_df['domain'] == domain) & (
                metric_df['instances'] == 1)][metric_perf].median()
            if bench in ['llama', 'yolo']:
                res['perf%'].append((perf_baseline / perf)*100)
            else:
                res['perf%'].append((perf / perf_baseline)*100)
            res['power'].append(power)
            power_baseline = platform_df.loc[(platform_df['domain'] == domain) & (
                platform_df['bench'] == bench) & (platform_df['instances'] == '1')]['SMI_power.draw'].median()
            res['power%'].append((power/power_baseline)*100)
            res['instances'].append(instances)
    result = pd.DataFrame(res).pivot_table(index=['instances'], values=[
        'perf', 'perf%', 'power', 'power%'], aggfunc='mean').reset_index()
    return result


def correlate_perf_power2(metric_df, platform_df, bench: str, metric_perf: str, baseline_platform, baseline_bench):
    res = {'domain': [], 'perf': [], 'perf%': [],
           'power': [], 'power%': [], 'instances': []}
    for domain in metric_df['domain'].unique():
        for instances in metric_df.loc[metric_df['domain'] == domain]['instances'].unique():
            perf = metric_df.loc[(metric_df['domain'] == domain) & (
                metric_df['instances'] == instances)][metric_perf].median()
            power = platform_df.loc[(platform_df['domain'] == domain) & (platform_df['bench'] == bench) & (
                platform_df['instances'] == str(instances))]['SMI_power.draw'].median() / instances
            res['domain'].append(domain)
            res['perf'].append(perf)
            perf_baseline = baseline_bench.loc[(baseline_bench['domain'] == domain) & (
                baseline_bench['instances'] == 1)][metric_perf].median()
            if bench in ['llama', 'yolo']:
                res['perf%'].append((perf_baseline / perf)*100)
            else:
                res['perf%'].append((perf / perf_baseline)*100)
            res['power'].append(power)
            power_baseline = baseline_platform.loc[(baseline_platform['domain'] == domain) & (
                baseline_platform['bench'] == bench) & (baseline_platform['instances'] == '1')]['SMI_power.draw'].median()
            res['power%'].append((power/power_baseline)*100)
            res['instances'].append(instances)
    # Pivot to get the mean between multiple GPUs
    result = pd.DataFrame(res).pivot_table(index=['instances'], values=[
        'perf', 'perf%', 'power', 'power%'], aggfunc='mean').reset_index()
    return result


dataset_domains = load_platform_df(
    file='data/250311-bench-muva-2xH100-pt1.csv')

root = 'data/'
bench_root = 'bench-res/'

files = {
    # V100-
    'V100-PCIE-32GB,250W': root + '250329-bench-chifflot-2xV100.csv',
    # A100
    'A100-SXM4-40GB,400W': root + '250326-bench-chuc-4xA100.csv',
    # H100
    'H100-NVL-94GB,400W': root + '250327-bench-muva-2xH100.csv',
}


all_platform_list = []
all_blender_list = []
all_hpcg_list = []
all_llama_list = []
all_yolo_list = []
all_blender_ratio_list = []
all_hpcg_ratio_list = []
all_llama_ratio_list = []
all_yolo_ratio_list = []

# '250323-migbench-chuc-4xA100-7g7.csv'
baseline_a100 = root + '250323-migbench-muva-2xH100-7g7.csv'
baseline_h100 = root + '250323-migbench-muva-2xH100-7g7.csv'
for label, locations in files.items():

    gpu, spec = label.split(',')

    # Load referential
    if 'A100' in label:
        baseline = baseline_a100
    else:
        baseline = baseline_h100
    baseline_platform = load_platform_df(baseline)
    baseline_blender = load_blender(baseline.replace(
        root, bench_root).replace('.csv', '-blender.csv'))
    baseline_hpcg = load_hpcg('data/250324-migbench-ovh-1xH100-7g7.csv'.replace(
        root, bench_root).replace('.csv', '-hpcg.csv'))
    baseline_llama = load_llama(baseline.replace(
        root, bench_root).replace('.csv', '-llama.csv'))
    baseline_yolo = load_yolo(baseline.replace(
        root, bench_root).replace('.csv', '-yolo.csv'))

    if isinstance(locations, str):
        locations = [locations]
    for location in locations:  # Iteration through location
        print('loading', location)

        # Load raw
        platform_df = load_platform_df(location)
        blender_df = load_blender(location.replace(
            root, bench_root).replace('.csv', '-blender.csv'))
        hpcg_df = load_hpcg(location.replace(
            root, bench_root).replace('.csv', '-hpcg.csv'))
        llama_df = load_llama(location.replace(
            root, bench_root).replace('.csv', '-llama.csv'))
        yolo_df = load_yolo(location.replace(
            root, bench_root).replace('.csv', '-yolo.csv'))

        platform_df['GPU'], platform_df['spec'] = gpu, spec
        all_platform_list.append(platform_df)

        if blender_df is not None:
            # Correlate Perf/Power
            # , baseline_platform, baseline_blender)
            blender_ratio_df = correlate_perf_power(
                blender_df, platform_df, 'blender', 'samples_per_minute')
            blender_df['GPU'], blender_df['spec'] = gpu, spec
            blender_ratio_df['GPU'], blender_ratio_df['spec'] = gpu, spec
            all_blender_list.append(blender_df)
            all_blender_ratio_list.append(blender_ratio_df)
        else:
            if 'pt2' not in location:
                print('Warning, no blender with', location)

        if hpcg_df is not None:
            # Correlate Perf/Power
            # , baseline_platform, baseline_hpcg)
            hpcg_ratio_df = correlate_perf_power(
                hpcg_df, platform_df, 'hpcg', 'GFLOP/s_Total_with_convergence_and_optimization_phase_overhead')
            hpcg_df['GPU'], hpcg_df['spec'] = gpu, spec
            hpcg_ratio_df['GPU'], hpcg_ratio_df['spec'] = gpu, spec
            all_hpcg_list.append(hpcg_df)
            all_hpcg_ratio_list.append(hpcg_ratio_df)
        else:
            if 'pt2' not in location:
                print('Warning, no hpcg with', location)

        if llama_df is not None:
            # Correlate Perf/Power
            # , baseline_platform, baseline_llama)
            llama_ratio_df = correlate_perf_power(
                llama_df, platform_df, 'llama', 'measure')
            llama_df['GPU'], llama_df['spec'] = gpu, spec
            llama_ratio_df['GPU'], llama_ratio_df['spec'] = gpu, spec
            all_llama_list.append(llama_df)
            all_llama_ratio_list.append(llama_ratio_df)
        else:
            if 'pt1' not in location:
                print('Warning, no llama with', location)

        if yolo_df is not None:
            # Correlate Perf/Power
            # , baseline_platform, baseline_yolo)
            yolo_ratio_df = correlate_perf_power(
                yolo_df, platform_df, 'yolo', 'measure')
            yolo_df['GPU'], yolo_df['spec'] = gpu, spec
            yolo_ratio_df['GPU'], yolo_ratio_df['spec'] = gpu, spec
            all_yolo_list.append(yolo_df)
            all_yolo_ratio_list.append(yolo_ratio_df)
        else:
            if 'pt1' not in location:
                print('Warning, no yolo with', location)

all_platform_df = pd.concat(all_platform_list)
all_blender_df = pd.concat(all_blender_list)
all_hpcg_df = pd.concat(all_hpcg_list)
all_llama_df = pd.concat(all_llama_list)
all_yolo_df = pd.concat(all_yolo_list)
all_blender_ratio_df = pd.concat(all_blender_ratio_list)
all_hpcg_ratio_df = pd.concat(all_hpcg_ratio_list)
all_llama_ratio_df = pd.concat(all_llama_ratio_list)
all_yolo_ratio_df = pd.concat(all_yolo_ratio_list)

# Introduce complemetary metrics
all_llama_df['measure_s'] = all_llama_df['measure'] / 1000
all_yolo_df['measure_s'] = all_yolo_df['measure'] / 1000
all_llama_ratio_df['perf_s'] = all_llama_ratio_df['perf'] / 1000
all_yolo_ratio_df['perf_s'] = all_yolo_ratio_df['perf'] / 1000
all_llama_df['measure_f'] = 1 / \
    (all_llama_df['measure_s'] / 60)  # nbre of inference per mn
all_yolo_df['measure_f'] = 1 / \
    (all_yolo_df['measure_s'] / 3600)  # nbre of training per h
# nbre of inference per mn
all_llama_ratio_df['perf_f'] = 1 / (all_llama_ratio_df['perf_s'] / 60)
all_yolo_ratio_df['perf_f'] = 1 / \
    (all_yolo_ratio_df['perf_s'] / 3600)  # nbre of training per h

all_blender_ratio_df['bench'] = 'blender'
all_hpcg_ratio_df['bench'] = 'HPCG'
all_llama_ratio_df['bench'] = 'Llama'
all_yolo_ratio_df['bench'] = 'Yolo'

all_ratio = pd.concat([all_blender_ratio_df, all_hpcg_ratio_df,
                      all_llama_ratio_df, all_yolo_ratio_df])
print('Done!')


all_ratio = all_ratio.drop(all_ratio.loc[(all_ratio['GPU'].str.startswith('V100')) & (
    all_ratio['bench'] == 'Llama') & (all_ratio['instances'] >= 7)].index)
all_ratio = all_ratio.drop(all_ratio.loc[(all_ratio['GPU'].str.startswith('A100')) & (
    all_ratio['bench'] == 'Llama') & (all_ratio['instances'] >= 8)].index)

ax = sns.lineplot(data=all_ratio, x="power%", y="perf%", hue="bench",
                  style='GPU', estimator="min", markers=True, dashes=False)
ax.set_ylabel('Performance per container (%)')
ax.set_xlabel('Power per container (%)')
ax.invert_xaxis()
# sns.lineplot(x=[0,100], y=[0,100], ax=ax, color='black', alpha=0.4, linestyle='--')

plt.gcf().savefig('figures/TS-perf-power.pdf', bbox_inches='tight')
