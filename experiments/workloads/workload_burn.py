import subprocess
from .workload_agent import WorkloadAgent

class WorkloadBurn(WorkloadAgent):

    def __init__(self, container_runtime : str = 'docker', prefix : str = None, delay : int = 300):
        self.container_runtime = container_runtime
        self.prefix = prefix
        self.delay  = delay

    def workload(self, gpu_id : str):
        cmd = [self.container_runtime, 'run', '--rm', '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=' + gpu_id, 'gpu_burn', './gpu_burn', str(self.delay)]
        if self.prefix is not None: cmd.insert(0, self.prefix)
        return cmd