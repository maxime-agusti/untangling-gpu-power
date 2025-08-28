import subprocess
import re

class KubectlWrapper(object):
    def __init__(self, prefix_command: list = ['minikube', 'kubectl', '--']):
        self.prefix_command = prefix_command

    def set_kube_replicas_policy(self, replicas: int, namespace: str = 'gpu-operator', config_name: str = 'oversub-all'):
        config_yaml = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config_name}
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: {replicas}
"""
        process = subprocess.run(
            self.prefix_command + ['apply', '-n', namespace, '-f', '-'],
            input=config_yaml,
            text=True,
            capture_output=True
        )

        if process.returncode == 0:
            print('ConfigMap updated successfully.')
        else:
            print('Error updating ConfigMap:', process.stderr)

    def patch_cluster_policy(self, namespace: str = 'gpu-operator', policy_name: str = 'cluster-policy', config_name: str = 'oversub-all-2', default_value: str = 'any'):
        patch_data = f'{{"spec": {{"devicePlugin": {{"config": {{"name": "{config_name}", "default": "{default_value}"}}}}}}}}'
        process = subprocess.run(
            self.prefix_command + ['patch', f'clusterpolicies.nvidia.com/{policy_name}', '-n', namespace, '--type', 'merge', '-p', patch_data],
            text=True,
            capture_output=True
        )

        if process.returncode == 0:
            print('Cluster policy patched successfully.')
        else:
            print('Error patching cluster policy:', process.stderr)

    def get_current_oversub_policy(self):
        process = subprocess.run(
            self.prefix_command + ['describe', 'nodes'],
            text=True,
            capture_output=True
        )
        if process.returncode != 0:
            print('Error retrieving node description:', process.stderr)
            return None

        match = re.search(r'nvidia.com/gpu\.replicas=(\d+)', process.stdout)
        if match:
            return int(match.group(1))

        print('No replicas information found.')
        return None

    def get_gpu_instance_count(self):
        process = subprocess.run(
            self.prefix_command + ['describe', 'nodes'],
            text=True,
            capture_output=True
        )
        if process.returncode != 0:
            print('Error retrieving node description:', process.stderr)
            return None

        match = re.search(r'nvidia.com/gpu:\s+(\d+)', process.stdout)
        if match:
            return int(match.group(1))

        print('No GPU instance information found.')
        return None

    def launch_pods(self, num_pods: int, image: str = 'gpu_burn', command: list = ['./gpu_burn', '-m', '10%', '3600'], namespace: str = 'default', pod_yaml_extension : str = ''):
        if num_pods <= 0: return
        pod_yaml = ''
        for i in range(num_pods):
            pod_name = f'{image}-{i}'
            # Delete the pod if it already exists
            subprocess.run(
                self.prefix_command + ['delete', 'pod', pod_name, '-n', namespace, '--ignore-not-found'],
                text=True,
                capture_output=True
            )

            pod_yaml += f"""---
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
spec:
  restartPolicy: Never
  containers:
  - name: {pod_name}
    image: {image}
    imagePullPolicy: Never
    command: {command}
    resources:
      limits:
        nvidia.com/gpu: 1
"""
        # Apply the updated YAML to create new pods
        process = subprocess.run(
            self.prefix_command + ['apply', '-n', namespace, '-f', '-'],
            input=pod_yaml,
            text=True,
            capture_output=True
        )
        if process.returncode == 0:
            print(f'{num_pods} pods launched successfully.')
        else:
            print('Error launching pods:', process.stderr)

    def launch_pods_blender(self, num_pods: int, namespace: str = 'default', label = 'default', result_directory : str = '/tmp'):
        command = ['python3', 'blender.py', 'results/blender.csv', label, './benchmark-launcher-cli', '--blender-version=4.3.0', '--device-type=CUDA', '--verbosity=0', 'benchmark', 'monster', '--json']
        pod_yaml_extension = f"""---
    volumeMounts:
    - name: blender-results
      mountPath: /app/results
  volumes:
  - name: blender-results
    hostPath:
      path: {result_directory}
      type: Directory
"""
        self.launch_pods(num_pods=num_pods, image='blender', command=command, namespace=namespace, pod_yaml_extension=pod_yaml_extension)

    def launch_pods_llama(self, num_pods: int, namespace: str = 'default', label = 'default', model_name : str = 'meta-llama/llama-3.2-1b', result_directory : str = '/tmp', cache_directory : str = '/home/root/.cache/huggingface/'):
        command = ['python3', 'llama.py', 'results/llama.csv', label, model_name]
        pod_yaml_extension = f"""---
    volumeMounts:
    - name: huggingface-cache
      mountPath: /root/.cache/huggingface/
    - name: llama-results
      mountPath: /app/results
  volumes:
  - name: huggingface-cache
    hostPath:
      path: {cache_directory}
      type: Directory
  - name: llama-results
    hostPath:
      path: {result_directory}
      type: Directory
"""
        self.launch_pods(num_pods=num_pods, image='llama', command=command, namespace=namespace, pod_yaml_extension=pod_yaml_extension)

    def launch_pods_hpcg(self, num_pods: int, namespace: str = 'default', label = 'default', result_directory : str = '/tmp'):
        command = ['python3', 'hpcg.py', 'results/hpcg.csv', label, './hpcg.sh', '--dat', 'custom-hpcg.dat']
        pod_yaml_extension = f"""---
    volumeMounts:
    - name: hpcg-results
      mountPath: /workspace/results
  volumes:
  - name: hpcg-results
    hostPath:
      path: {result_directory}
      type: Directory
"""
        self.launch_pods(num_pods=num_pods, image='hpcg', command=command, namespace=namespace, pod_yaml_extension=pod_yaml_extension)

    def launch_pods_yolo(self, num_pods: int, namespace: str = 'default', label = 'default', model_name : str = 'yolov8n.pt', result_directory : str = '/tmp'):
        command = ['python3', 'yolo.py', 'results/yolo.csv', label, model_name]
        pod_yaml_extension = f"""---
    args:
    - "--shm-size=4g"
    volumeMounts:
    - name: yolo-results
      mountPath: /app/results
  volumes:
  - name: yolo-results
    hostPath:
      path: {result_directory}
      type: Directory
"""
        self.launch_pods(num_pods=num_pods, image='yolo', command=command, namespace=namespace, pod_yaml_extension=pod_yaml_extension)

    def destroy_all_pods(self, namespace: str = 'default'):
        process = subprocess.run(
            self.prefix_command + ['delete', 'pods', '--all', '-n', namespace],
            text=True,
            capture_output=True
        )

        if process.returncode == 0:
            print('All pods deleted successfully.')
        else:
            print('Error deleting pods:', process.stderr)
