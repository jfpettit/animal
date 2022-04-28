from tensorboardX import SummaryWriter
import os
import yaml

class TensorboardLogger:
    def __init__(self, run_name: str = None, folder_name: str = "tensorboards"):
        self.folder = os.path.join(os.getcwd(), folder_name, run_name)
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder, exist_ok=True)

        self.writer = SummaryWriter(logdir=self.folder)

    def save_config(self, config, name):
        config_path = self.folder + f"/{name}.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

    def log(self, name, data, step=None):
        self.writer.add_scalar(name, data, global_step=step)

    def logdict(self, data, step=None):
        for k, v in data.items():
            self.log(k, v, step=step)
