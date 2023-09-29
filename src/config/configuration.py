from pathlib import Path
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import yaml
from box.exceptions import BoxValueError
from dataclasses import dataclass

CONFIG_FILE_PATH = "./config.yml"

@dataclass(frozen=True)
class PrepareModelConfig:
    model_path: Path
    model_class_indexes_path: Path

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    """
    This function reads a yaml file and returns a ConfigBox object. 

    Args:
        path_to_yaml (Path): path to yaml file.

    Raises:
        ValueError: if yaml file is empty.
        e: if any other error occurs.
    
    Returns:
        ConfigBox: ConfigBox object.
    """
    try: 
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty.")
    except Exception as e:
        raise e  

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        config_filepath = Path(config_filepath)

        self.config = read_yaml(config_filepath)
    
    
    def get_prepare_densenet_model_config(self) -> PrepareModelConfig:
        config_base = self.config.densenet
        
        prepare_base_model_config = PrepareModelConfig(
            model_path = config_base.model_path,
            model_class_indexes_path = config_base.classes_indexes_path
        )
        
        return prepare_base_model_config
    
    def get_prepare_vgg19_mri_model_config(self) -> PrepareModelConfig:
        config_mri = self.config.vgg19
        
        prepare_base_model_config = PrepareModelConfig(
            model_path = config_mri.model_path,
            model_class_indexes_path = config_mri.classes_indexes_path
        )
        
        return prepare_base_model_config