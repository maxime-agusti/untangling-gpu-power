from monitoring import *
from gpu_settings import *
from workloads import *

import sys, time, re

DELAY=1800

# This script evaluates performance under GPU time-slicing by running multiple containers on the same GPU.
# No MIG configuration is used — instead, workload concurrency is increased by launching more containers.
# Benchmarks are run in each container, and metrics (DCGM, IPMI, CPU) are collected to study sharing impact.

#########################
# Launch bench          #
#########################
def setup_and_launch(mig_wrapper, docker_wrapper, monitors_wrapper, gpu_count):

    gpu_uuid_list =  mig_wrapper.list_gpu_uuid()
    docker_wrapper.destroy_all_containers()
    oversub = 8 
    # Use different workloads
    for bench in ['blender', 'hpcg', 'llama', 'yolo']:

        # II) Iterate through different number of pods
        for instance_per_gpu in range(1,oversub+1):
            
            # III) Update monitoring
            setting_name = bench + '|' + str(instance_per_gpu)
            monitors_wrapper.update_monitoring({'context': setting_name}, monitor_index=0, reset_launch=True)
            print(setting_name)

            for gpu_index, gpu_uuid in enumerate(gpu_uuid_list):
                for _ in range(instance_per_gpu):
                    launch_container_from_bench(bench, gpu_uuid=gpu_uuid, label=setting_name+'|'+str(gpu_index))
            time.sleep(DELAY + 10)

            # IV) Clean up
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
    monitors_wrapper = MonitorWrapper(monitors=monitors, output_file='measures-perf-timeslices.csv')

    ##########################
    # Starting  measurements #
    ##########################

    try:
        monitors_wrapper.start_monitoring()

        print('Capturing idle')
        monitors_wrapper.update_monitoring({'context':'idle'}, monitor_index=0, reset_launch=False)
        time.sleep(DELAY)
        print('Idle capture ended')

        setup_and_launch(mig_wrapper, docker_wrapper, monitors_wrapper, gpu_count)

    except KeyboardInterrupt:
        pass
    print('Exiting')
    monitors_wrapper.stop_monitoring()