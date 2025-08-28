import subprocess
from .workload_agent import WorkloadAgent

class WorkloadHpcg(WorkloadAgent):

    def __init__(self, container_runtime : str = 'docker', prefix : str = None):
        self.container_runtime = container_runtime
        self.prefix = prefix

    # docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
    def workload(self, gpu_id : str, label : str = 'default', result_directory : str = ''):
        cmd = [self.container_runtime, 'run', '--rm', '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=' + gpu_id,
               '--ipc=host', '--ulimit', 'memlock=-1',  '--ulimit', 'stack=67108864',
               '-v', result_directory + ':/workspace/results', 'hpcg',
               'python3', 'hpcg.py', 'results/hpcg.csv', label, './hpcg.sh', '--dat', 'custom-hpcg.dat'
            ]
        if self.prefix is not None: cmd.insert(0, self.prefix)
        return cmd