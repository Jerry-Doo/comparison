import os
import yaml

def load_yaml_config(config_name):
    # 获取当前文件的上两级目录（即 recurrent_tof_denoising_pytorch/）
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_dir = os.path.join(root_dir, 'configs')
    config_path = os.path.join(config_dir, config_name)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)