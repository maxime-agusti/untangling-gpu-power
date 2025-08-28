from monitoring import *
from gpu_settings import *
from workloads import *

import sys, time, re

DELAY=300

# This script evaluates worst-case power consumption under GPU time-slicing using GPU burn workloads.
# No benchmarks are used — the focus is solely on stressing the GPU with multiple GPU burn containers.
# Metrics (especially power via DCGM and IPMI) are collected to assess peak consumption under contention.

#########################
# Setup Replicas policy #
#########################
def setup_namespace_and_launch(kubectl_wrapper, monitors_wrapper, gpu_count):

    kubectl_wrapper.destroy_all_pods()

    oversub_list = [1,2,4,8] # 1 must be first as we use in that case the default setting
    for oversub in oversub_list:

        # I) Setup oversubscription policy
        if oversub > 1:
            kubectl_wrapper.set_kube_replicas_policy(oversub, config_name="oversub-all-" + str(oversub))
            kubectl_wrapper.patch_cluster_policy(config_name="oversub-all-" + str(oversub))
            while(True):
                current_value = kubectl_wrapper.get_current_oversub_policy()
                if current_value == oversub: break
                time.sleep(5) # Waiting for patch to be applied, can be long

        print("New GPU instance count:", kubectl_wrapper.get_gpu_instance_count())

        # II) Iterate through different number of pods
        for instance_per_gpu in range(oversub+1):

            wanted_instance = instance_per_gpu * gpu_count
            
            # III) Update monitoring
            setting_name = str(oversub) + '|' + str(instance_per_gpu)
            monitors_wrapper.update_monitoring({'context': setting_name}, monitor_index=0, reset_launch=True)
            print(setting_name)

            kubectl_wrapper.launch_pods(num_pods=wanted_instance, command=["./gpu_burn", "-m", "10%", str(DELAY)])
            time.sleep(DELAY + 10)

            # IV) Clean up
            kubectl_wrapper.destroy_all_pods()
            time.sleep(10)

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
    kubectl_wrapper = KubectlWrapper()

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

        setup_namespace_and_launch(kubectl_wrapper, monitors_wrapper, gpu_count)

    except KeyboardInterrupt:
        pass
    print('Exiting')
    monitors_wrapper.stop_monitoring()