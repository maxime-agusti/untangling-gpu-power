import subprocess
from .workload_agent import WorkloadAgent

class WorkloadTrainingYolo(WorkloadAgent):

    def __init__(self, container_runtime : str = 'docker', prefix : str = None):
        self.container_runtime = container_runtime
        self.prefix = prefix

    def workload(self, gpu_id : str, label : str = 'default', model_name : str = 'yolov8n.pt', result_directory : str = ''):
        cmd = [self.container_runtime, 'run', '--rm', '--runtime=nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=' + gpu_id,
               '-v', result_directory + ':/app/results', '--shm-size=4g', 'yolo',
               'python3', 'yolo.py', 'results/yolo.csv', label, model_name
            ]
        if self.prefix is not None: cmd.insert(0, self.prefix)
        return cmd