from monitoring import *
from gpu_settings import *
from workloads import *

import sys, time, re

DELAY=1800
# This script analyzes how different GPU Instance (GI) sizes impact performance and system metrics using real-world benchmarks.
# It runs workloads like Blender, HPCG, LLaMA, and YOLO, without modifying Compute Instances (CI size = GI size).
# Metrics from DCGM, IPMI, and CPU are collected to study the link between performance and energy/resource usage.


#########################
# Select MIG instances  #
#########################
def select_gi_and_launch(mig_wrapper, docker_wrapper, monitors_wrapper, suitable_gpus):
    gi_profile_list = mig_wrapper.list_gpu_instance_profiles(gpu_id=suitable_gpus[0]) # We assume homogeneity on GPUs
    gi_profile = gi_profile_list[0] # GI size under scrutiny

    if gi_profile['free_instances'] <= 0:
        print('Error, not enough free instances on profile ', gi_profile)
        sys.exit(-1)

    print('Creating', gi_profile['name'], 'on all GPUs')
    iterate_on_complements(mig_wrapper, docker_wrapper, monitors_wrapper, suitable_gpus, gi_profile['name'])

#########################
#Complementary instances#
#########################
def iterate_on_complements(mig_wrapper, docker_wrapper, monitors_wrapper, suitable_gpus, gi_profile):
      
    index=0
    while True:
        index+=1
        stop_signal = False
        # Test if we can create a new GI on all GPUs
        for suitable_gpu in suitable_gpus:
            gi = [gi for gi in mig_wrapper.list_gpu_instance_profiles(gpu_id=suitable_gpu) if gi['name'] == gi_profile][0] # only one result matching condition
            if gi['free_instances'] > 0:
                mig_wrapper.create_gpu_instance(gpu_id=suitable_gpu, gi_profiles=gi_profile, create_ci=True)
            else:
                stop_signal = True
                break

        if stop_signal:
            break

        launch_bench(mig_wrapper, docker_wrapper, monitors_wrapper, ci_count_per_gpu=index)

    mig_wrapper.destroy_compute_instance()
    mig_wrapper.destroy_gpu_instance()

#########################
# Launch bench          #
#########################
def launch_bench(mig_wrapper, docker_wrapper, monitors_wrapper, ci_count_per_gpu : int):

    gpu_uuid_list = mig_wrapper.list_usable_mig_partition()
    docker_wrapper.destroy_all_containers()
    # Use different workloads
    for bench in ['blender', 'hpcg', 'llama', 'yolo']:

        # I) Update monitoring
        setting_name = bench + '|' + str(ci_count_per_gpu)
        monitors_wrapper.update_monitoring({'context': setting_name}, monitor_index=0, reset_launch=True)

         # II) Launch
        for ci_index, gpu_uuid in enumerate(gpu_uuid_list):

            gpu_index = int(ci_index / ci_count_per_gpu) # GPU index can be deducted like this on HOMOGENEOUS profiles per GPU
            launch_container_from_bench(bench, gpu_uuid=gpu_uuid['mig_uuid'], label=setting_name+'|'+str(gpu_index))

        time.sleep(DELAY + 10)

        # III) Clean
        docker_wrapper.destroy_all_containers()
        time.sleep(10)

def launch_container_from_bench(bench : str, gpu_uuid : str, label : str):
    result_directory = '/home/pjacquet/cloud-gpu-manager/bench-res'
    cache_directory  = '/home/pjacquet/.cache/huggingface/'
    if bench == 'blender':
        my_workload = WorkloadBlender()
        my_workload.run(gpu_id=gpu_uuid, label=label, result_directory=result_directory)
    elif bench == 'hpcg':
        my_workload = WorkloadHpcg()
        my_workload.run(gpu_id=gpu_uuid, label=label, result_directory=result_directory)
    elif bench == 'llama':
        my_workload = WorkloadInferenceLlama()
        my_workload.run(gpu_id=gpu_uuid, label=label, result_directory=result_directory, cache_directory=cache_directory)
    elif bench == 'yolo':
        my_workload = WorkloadTrainingYolo()
        my_workload.run(gpu_id=gpu_uuid, label=label, result_directory=result_directory)
    else:
        print('Unknow bench specified')
        sys.exit(-1)

if __name__ == "__main__":

    print('Starting time-slice experiment')

    #########################
    # Hardware management   #
    #########################
    mig_wrapper = MIGWrapper(sudo_command='sudo')
    gpu_count = mig_wrapper.gpu_count()
    if gpu_count <= 0:
        print('Not enough GPU to continue')
        sys.exit(-1)

    docker_wrapper = DockerWrapper()

    ##########################
    # Monitoring management  #
    ##########################
    mon_labels = ConstMonitor({'context':'init'}, gpu_count=gpu_count, include_gpu_x=True)
    mon_cpu = CPUMonitor(gpu_count=gpu_count, include_gpu_x=True)
    mon_ipmi = IPMIMonitor(sudo_command='sudo')
    mon_ipmi.discover()
    mon_smi  = SMIMonitor(sudo_command='sudo')
    mon_dcgm = DCGMMonitor(url='http://localhost:9400/metrics')

    monitors = [mon_labels, mon_cpu, mon_ipmi, mon_smi, mon_dcgm] # index matters for update
    monitors_wrapper = MonitorWrapper(monitors=monitors, output_file='measures-perf-mig-gi.csv')

    ##########################
    # Setup experiment       #
    ##########################
    mig_wrapper.clean_reset()

    suitable_gpus = list() # List of gpu id
    mig_status = mig_wrapper.check_mig_status()
    for mig_gpu, status in enumerate(mig_status):
        active, _ = status
        if active: suitable_gpus.append(mig_gpu)
    print('List of GPUs with MIG currently operational:', suitable_gpus)
    if not suitable_gpus:
        print('Not enough MIG hardware to continue')
        sys.exit(-1)

    ##########################
    # Starting  measurements #
    ##########################

    try:
        monitors_wrapper.start_monitoring()

        print('Capturing idle')
        monitors_wrapper.update_monitoring({'context':'idle'}, monitor_index=0, reset_launch=False)
        time.sleep(DELAY)
        print('Idle capture ended')

        select_gi_and_launch(mig_wrapper, docker_wrapper, monitors_wrapper, suitable_gpus)

    except KeyboardInterrupt:
        pass
    print('Exiting')
    monitors_wrapper.stop_monitoring()
