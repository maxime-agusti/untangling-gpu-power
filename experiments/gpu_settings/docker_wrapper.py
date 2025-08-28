import subprocess

class DockerWrapper(object):

    def __init__(self, prefix_command: list = ['docker']):
        self.prefix_command = prefix_command

    def destroy_all_containers(self, exclude_container_name: str = "dcgm-exporter"):
        try:
            # Get container IDs and names in one command
            container_info = subprocess.check_output(self.prefix_command + ["ps", '-a', "--format", "{{.ID}} {{.Names}}"], text=True).strip()

            if not container_info:
                print("No containers found.")
                return

            # Parse container info and filter out the excluded container
            containers = [line.split()[0] for line in container_info.split("\n") if line and line.split()[1] != exclude_container_name]

            if containers:
                # Stop and remove the filtered containers
                subprocess.run(self.prefix_command + ["stop"] + containers, check=True)
                subprocess.run(self.prefix_command + ["rm"] + containers, check=True)
                print("All containers (except '{}') have been stopped and removed.".format(exclude_container_name))
            else:
                print("No containers to remove (or only '{}' is running).".format(exclude_container_name))

        except subprocess.CalledProcessError as e:
            print("Error managing containers:", e)