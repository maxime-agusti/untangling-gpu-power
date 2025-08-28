import subprocess
from .workload_agent import WorkloadAgent

class WorkloadInferenceLlama(WorkloadAgent):

    def __init__(self, container_runtime : str = 'docker', prefix : str = None):
        self.container_runtime = container_runtime
        self.prefix = prefix

    def workload(self, gpu_id : str, label : str = 'default', model_name : str = 'meta-llama/llama-3.2-1b', result_directory : str = '', cache_directory : str = ''):
        cmd = [self.container_runtime, 'run', '--rm', '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=' + gpu_id,
               '-v', cache_directory + ':/root/.cache/huggingface/', '-v', result_directory + ':/app/results', 'llama',
               'python3', 'llama.py', 'results/llama.csv', label, model_name
            ]
        if self.prefix is not None: cmd.insert(0, self.prefix)
        return cmd