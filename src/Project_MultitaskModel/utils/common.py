import os
from box.exceptions import BoxValueError
import yaml
from src.Project_MultitaskModel import logger
import json
import torch
from ensure import ensure_annotations
from box import ConfigBox
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    try :
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise e
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory at: {path}")
        
        
@ensure_annotations
def save_json(path: str, data: dict) -> None:
    with open (path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"JSON file saved at: {path}")
    
@ensure_annotations
def load_json(path: str) -> ConfigBox:
    with open (path, 'r') as json_file:
        content = json.load(json_file)
    logger.info(f"JSON file loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(path: str, data: bytes) -> None:
    with open(path, 'wb') as bin_file:
        bin_file.write(data)
    logger.info(f"Binary file saved at: {path}")
    

@ensure_annotations
def load_bin(path: str) -> bytes:
    with open(path, 'rb') as bin_file:
        data = bin_file.read()
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def decode_base64_to_image(base64_string: str, output_path: str) -> None:
    image_data = base64.b64decode(base64_string)
    with open(output_path, 'wb') as image_file:
        image_file.write(image_data)
        image_file.close()
    logger.info(f"Image decoded from base64 and saved at: {output_path}")
    
@ensure_annotations
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read())