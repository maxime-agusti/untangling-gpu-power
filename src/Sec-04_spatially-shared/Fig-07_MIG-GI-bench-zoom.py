import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.color_palette("Set2")
sns.set_theme(style="darkgrid")


def load_platform_df(file: str):

    df = pd.read_csv(file)
    df.reset_index(inplace=True)

    timestamp_decrease = df.timestamp.diff() < 0
    restart = timestamp_decrease[timestamp_decrease].index.values
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


def correlate_perf_power(metric_df, platform_df, bench: str, metric_perf: str, baseline_platform, baseline_bench):
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


all_platform_list = []
all_blender_list = []
all_hpcg_list = []
all_llama_list = []
all_yolo_list = []
all_blender_ratio_list = []
all_hpcg_ratio_list = []
all_llama_ratio_list = []
all_yolo_ratio_list = []


root = 'data/'
bench_root = 'bench-res/'

files = {
    # A100-PCIE-40GB (250W) | Vary CI sise but not GI (GI == max)
    '1g,A100-PCIE-40GB (250W),ci': (root + '250322-migbench-grouille-2xA100-7g1-pt1.csv', root + '250320-migbench-grouille-2xA100-7g1-pt2.csv'),
    '2g,A100-PCIE-40GB (250W),ci': root + '250323-migbench-grouille-2xA100-7g2.csv',
    '3g,A100-PCIE-40GB (250W),ci': root + '250323-migbench-grouille-2xA100-7g3.csv',
    '4g,A100-PCIE-40GB (250W),ci': root + '250324-migbench-grouille-2xA100-7g4.csv',
    '7g,A100-PCIE-40GB (250W),ci': root + '250324-migbench-grouille-2xA100-7g7.csv',
    # A100-PCIE-80GB (300W) | Vary CI sise but not GI (GI == max)
    '1g,A100-PCIE-80GB (300W),ci': root + '250321-migbench-ovh-1xA100-7g1.csv',
    '2g,A100-PCIE-80GB (300W),ci': root + '250322-migbench-ovh-1xA100-7g2.csv',
    '3g,A100-PCIE-80GB (300W),ci': root + '250323-migbench-ovh-1xA100-7g3.csv',
    '4g,A100-PCIE-80GB (300W),ci': root + '250323-migbench-ovh-1xA100-7g4.csv',
    '7g,A100-PCIE-80GB (300W),ci': root + '250323-migbench-ovh-1xA100-7g7.csv',
    # A100-SXM4-40GB (400W) | Vary CI sise but not GI (GI == max)
    '1g,A100-SXM4-40GB (400W),ci': root + '250321-migbench-chuc-4xA100-7g1.csv',
    '2g,A100-SXM4-40GB (400W),ci': root + '250322-migbench-chuc-4xA100-7g2.csv',
    '3g,A100-SXM4-40GB (400W),ci': root + '250323-migbench-chuc-4xA100-7g3.csv',
    '4g,A100-SXM4-40GB (400W),ci': root + '250323-migbench-chuc-4xA100-7g4.csv',
    '7g,A100-SXM4-40GB (400W),ci': root + '250323-migbench-chuc-4xA100-7g7.csv',
    # H100-PCIE-80GB (350W) | Vary CI sise but not GI (GI == max)
    '1g,H100-PCIE-80GB (350W),ci': root + '250322-migbench-ovh-1xH100-7g1.csv',
    '2g,H100-PCIE-80GB (350W),ci': root + '250323-migbench-ovh-1xH100-7g2.csv',
    '3g,H100-PCIE-80GB (350W),ci': root + '250323-migbench-ovh-1xH100-7g3.csv',
    '4g,H100-PCIE-80GB (350W),ci': root + '250323-migbench-ovh-1xH100-7g4.csv',
    '7g,H100-PCIE-80GB (350W),ci': root + '250324-migbench-ovh-1xH100-7g7.csv',
    # H100-NVL-94GB (400W)  | Vary CI sise but not GI (GI == max)
    # Also available 250324-migbench-muva-2xH100-7g1-x2sm.csv
    '1g,H100-NVL-94GB (400W),ci': root + '250324-migbench-muva-2xH100-7g1-x2sm.csv',
    '2g,H100-NVL-94GB (400W),ci': root + '250325-migbench-muva-2xH100-7g2.csv',
    '3g,H100-NVL-94GB (400W),ci': (root + '250325-migbench-muva-2xH100-7g3-pt1.csv', root + '250325-migbench-muva-2xH100-7g3-pt2.csv'),
    '4g,H100-NVL-94GB (400W),ci': root + '250324-migbench-muva-2xH100-7g4.csv',
    '7g,H100-NVL-94GB (400W),ci': root + '250323-migbench-muva-2xH100-7g7.csv',
}

# '250323-migbench-chuc-4xA100-7g7.csv'
baseline_a100 = root + '250323-migbench-muva-2xH100-7g7.csv'
baseline_h100 = root + '250323-migbench-muva-2xH100-7g7.csv'
for label, locations in files.items():

    compute_size, gpu, splitting = label.split(',')

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

        platform_df['compute_size'], platform_df['GPU'], platform_df['splitting'] = compute_size, gpu, splitting
        all_platform_list.append(platform_df)

        if blender_df is not None:
            # Correlate Perf/Power
            blender_ratio_df = correlate_perf_power(
                blender_df, platform_df, 'blender', 'samples_per_minute', baseline_platform, baseline_blender)
            blender_df['compute_size'], blender_df['GPU'], blender_df['splitting'] = compute_size, gpu, splitting
            blender_ratio_df['compute_size'], blender_ratio_df['GPU'], blender_ratio_df['splitting'] = compute_size, gpu, splitting
            all_blender_list.append(blender_df)
            all_blender_ratio_list.append(blender_ratio_df)
        else:
            if 'pt2' not in location:
                print('Warning, no blender with', location)

        if hpcg_df is not None:
            # Correlate Perf/Power
            hpcg_ratio_df = correlate_perf_power(
                hpcg_df, platform_df, 'hpcg', 'GFLOP/s_Total_with_convergence_and_optimization_phase_overhead', baseline_platform, baseline_hpcg)
            hpcg_df['compute_size'], hpcg_df['GPU'], hpcg_df['splitting'] = compute_size, gpu, splitting
            hpcg_ratio_df['compute_size'], hpcg_ratio_df['GPU'], hpcg_ratio_df['splitting'] = compute_size, gpu, splitting
            all_hpcg_list.append(hpcg_df)
            all_hpcg_ratio_list.append(hpcg_ratio_df)
        else:
            if 'pt2' not in location:
                print('Warning, no hpcg with', location)

        if llama_df is not None:
            # Correlate Perf/Power
            llama_ratio_df = correlate_perf_power(
                llama_df, platform_df, 'llama', 'measure', baseline_platform, baseline_llama)
            llama_df['compute_size'], llama_df['GPU'], llama_df['splitting'] = compute_size, gpu, splitting
            llama_ratio_df['compute_size'], llama_ratio_df['GPU'], llama_ratio_df['splitting'] = compute_size, gpu, splitting
            all_llama_list.append(llama_df)
            all_llama_ratio_list.append(llama_ratio_df)
        else:
            if 'pt1' not in location:
                print('Warning, no llama with', location)

        if yolo_df is not None:
            # Correlate Perf/Power
            yolo_ratio_df = correlate_perf_power(
                yolo_df, platform_df, 'yolo', 'measure', baseline_platform, baseline_yolo)
            yolo_df['compute_size'], yolo_df['GPU'], yolo_df['splitting'] = compute_size, gpu, splitting
            yolo_ratio_df['compute_size'], yolo_ratio_df['GPU'], yolo_ratio_df['splitting'] = compute_size, gpu, splitting
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

all_blender_ratio_df['gi_sum'] = all_blender_ratio_df['compute_size'].str.replace(
    'g', '').astype(int) * all_blender_ratio_df['instances']
all_hpcg_ratio_df['gi_sum'] = all_hpcg_ratio_df['compute_size'].str.replace(
    'g', '').astype(int) * all_hpcg_ratio_df['instances']
all_llama_ratio_df['gi_sum'] = all_llama_ratio_df['compute_size'].str.replace(
    'g', '').astype(int) * all_llama_ratio_df['instances']
all_yolo_ratio_df['gi_sum'] = all_yolo_ratio_df['compute_size'].str.replace(
    'g', '').astype(int) * all_yolo_ratio_df['instances']

all_platform_df['gi_sum'] = all_platform_df['compute_size'].str.replace(
    'g', '').astype(int) * all_platform_df['instances'].astype(int)
all_blender_df['gi_sum'] = all_blender_df['compute_size'].str.replace(
    'g', '').astype(int) * all_blender_df['instances']
all_hpcg_df['gi_sum'] = all_hpcg_df['compute_size'].str.replace(
    'g', '').astype(int) * all_hpcg_df['instances']
all_llama_df['gi_sum'] = all_llama_df['compute_size'].str.replace(
    'g', '').astype(int) * all_llama_df['instances']
all_yolo_df['gi_sum'] = all_yolo_df['compute_size'].str.replace(
    'g', '').astype(int) * all_yolo_df['instances']
print('Done!')

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

GPU = 'H100-NVL-94GB (400W)'

df_1g_blender = all_blender_ratio_df.loc[(
    all_blender_ratio_df['compute_size'] == '1g') & (all_blender_ratio_df['GPU'] == GPU)]
df_1g_blender['bench'] = 'blender'
df_1g_hpcg = all_hpcg_ratio_df.loc[(
    all_hpcg_ratio_df['compute_size'] == '1g') & (all_hpcg_ratio_df['GPU'] == GPU)]
df_1g_hpcg['bench'] = 'hpcg'
df_1g_llama = all_llama_ratio_df.loc[(
    all_llama_ratio_df['compute_size'] == '1g') & (all_llama_ratio_df['GPU'] == GPU)]
df_1g_llama['bench'] = 'llama'
df_1g_yolo = all_yolo_ratio_df.loc[(
    all_yolo_ratio_df['compute_size'] == '1g') & (all_yolo_ratio_df['GPU'] == GPU)]
df_1g_yolo['bench'] = 'yolo'

df_1g = pd.concat([df_1g_blender, df_1g_hpcg, df_1g_llama, df_1g_yolo])
sns.lineplot(
    data=df_1g, x="gi_sum", y="power%",
    hue="bench", style="bench",
    markers={"blender": "o", "hpcg": "s", "llama": "^",
             "yolo": "X"},  # map each bench to a marker
    dashes=False, markersize=8,
    ax=axes[0]
)
sns.lineplot(
    data=df_1g, x="gi_sum", y="perf%",
    hue="bench", style="bench",
    markers={"blender": "o", "hpcg": "s", "llama": "^",
             "yolo": "X"},  # map each bench to a marker
    dashes=False, markersize=8,
    ax=axes[1]
)
axes[0].set_xlabel("Compute slice allocated", fontsize=14)
axes[0].set_ylabel("Power per container (%)", fontsize=14)
axes[1].get_legend().remove()
axes[1].set_xlabel("Compute slice allocated", fontsize=14)
axes[1].set_ylabel("Performance per container (%)", fontsize=14)

# fig.text(0.5, -0.05, "Scaling with number of 1g slices on H100-NVL-94GB (power & performance)", ha="center")
plt.gcf().savefig('figures/MIG-GI-bench-zoom.pdf', bbox_inches='tight')
