import subprocess
from .workload_agent import WorkloadAgent

class WorkloadBlender(WorkloadAgent):

    def __init__(self, container_runtime : str = 'docker', prefix : str = None):
        self.container_runtime = container_runtime
        self.prefix = prefix

    def workload(self, gpu_id : str, label : str = 'default', result_directory : str = ''):
        cmd = [self.container_runtime, 'run', '--rm', '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=' + gpu_id,
               '-v', result_directory + ':/app/results', 'blender',
               'python3', 'blender.py', 'results/blender.csv', label, './benchmark-launcher-cli', '--blender-version=4.3.0', '--device-type=CUDA', '--verbosity=0', 'benchmark', 'monster', '--json'
            ] 
        if self.prefix is not None: cmd.insert(0, self.prefix)
        return cmd