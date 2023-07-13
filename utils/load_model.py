import os

import pandas as pd
import torch

from models.TestNets import TestNet, QTestNet


def load_model(model_type_str, model_source, model_id_str):
    if model_type_str.startswith('Q'):
        results_filename = f'Q_{model_id_str}'
    else:
        results_filename = f'{model_id_str}'
    model_results_file_path = os.path.join(model_type_str, model_source, results_filename)
    checkpoint_folder_path = os.path.join('CHECKPOINT_FOLDER', model_results_file_path)

    model_attributes_path = os.path.join(checkpoint_folder_path, 'model_attributes.csv')
    model_attributes = pd.read_csv(model_attributes_path)
    model_attributes = model_attributes.to_dict()

    if model_type_str == 'TestNet':
        model = TestNet(model_attributes['input_channels'], model_attributes['num_channels'],
                        model_attributes['num_layers'], model_attributes['kernel_size'], model_attributes['stride'],
                        model_attributes['source'], model_attributes['spec_output'])
    if model_type_str == 'QTestNet':
        model = QTestNet(model_attributes['input_channels'], model_attributes['num_channels'],
                        model_attributes['num_layers'], model_attributes['kernel_size'], model_attributes['stride'],
                        model_attributes['source'], model_attributes['spec_output'])

    # Load state dict
    checkpoint_files_list = os.listdir(checkpoint_folder_path)
    checkpoint_files_list.sort()
    checkpoint_file_name = checkpoint_files_list[-2]
    assert checkpoint_file_name.startswith('epoch')
    checkpoint_file_path = os.path.join(checkpoint_folder_path, checkpoint_file_name)
    model.load_state_dict(torch.load(checkpoint_file_path))
    return model