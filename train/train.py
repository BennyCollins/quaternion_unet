import os
import time
import gc
from datetime import datetime

import torch
import torch.nn as nn

from utils.utils import create_folder
from utils.plot_spec import plot_chunk_specs
from dataset.cac_dataloader import CACUnetInput
from dataset.complex_as_channel import ispectro, icomplex_as_channel
from models.TestNets import TestNet, QTestNet
from models.UNet import UNet
from models.QUNet import QUNet
from evaluate.evaluate import eval_model
from apply.apply_model import apply_model
from train.train_utils import get_model_size_mb, create_training_graphs
from settings import RESULTS_FOLDER, BATCH_SIZE, EPOCHS, LR, BETAS, WEIGHT_DECAY, TRAIN_STANDARD_TEST_MODEL, \
    TRAIN_STANDARD_MODEL, NUM_CHANNELS_STANDARD, NUM_LAYERS_STANDARD, TRAIN_QUATERNION_TEST_MODEL, \
    TRAIN_QUATERNION_MODEL, NUM_CHANNELS_QUATERNION, NUM_LAYERS_QUATERNION, TYPE, SPEC_OUTPUT, FRAMES_PER_SAMPLE, \
    HOP_LENGTH, REPORT_PER_N_BATCHES, FILTER_TRAIN_SET, FILTER_VAL_SET, EVAL_AFTER_TRAINING, EVAL_ON_TEST_SET, \
    EVAL_ON_TRAIN_SET, EVAL_ON_VAL_SET, APPLY_AFTER_TRAINING, USE_CUDA


'''
Training functions in this file are based on code from url below with large adaptations and additions:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
'''


def train_one_epoch(model, optimizer, loss_fn, training_loader, epoch_index, batch_loss_dict, report_per_n_batches,
                    device='cpu'):
    total_loss_over_epoch = 0
    running_loss = torch.zeros(report_per_n_batches, dtype=torch.float)
    running_loss[running_loss == 0] = float('nan')

    for i, data in enumerate(training_loader):

        input, target = data
        input, target = input.to(device), target.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(input)

        # Compute the loss and its gradients
        if model.spec_output:
            # Convert to audio
            output = icomplex_as_channel(output)
            output = ispectro(output, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)
        # Calculate loss
        loss = loss_fn(output, target)
        loss.backward()

        # Delete output to free up memory
        del output

        total_loss_over_epoch += loss.item()

        batch_number = epoch_index * len(training_loader) + i + 1
        batch_loss_dict['raw'][batch_number] = loss.item()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        loss_index = i % report_per_n_batches
        running_loss[loss_index] = loss.item()

        # Delete loss to free up memory
        del loss

        last_loss = torch.nanmean(running_loss,
                                  dtype=torch.float)  # Average loss over last report_per_n_batches batches
        if i % report_per_n_batches == report_per_n_batches - 1:
            print(f'Batch {i + 1} avg loss over previous {report_per_n_batches}: {last_loss.item()}')

        batch_loss_dict['avg_over_n_batches'][batch_number] = last_loss.item()
    avg_loss_over_epoch = total_loss_over_epoch / (i + 1)
    return avg_loss_over_epoch, batch_number


def train_model(model, lr, betas, weight_decay, max_epochs=100, min_epochs=30, early_stopping_patience=3, batch_size=8,
                loss='MSE', report_per_n_batches=20, filter_train=False, filter_val=False, eval_after_training=True,
                eval_on_test_set=True, eval_on_train_set=False, eval_on_val_set=False, apply_after_training=True,
                device='cpu'):

    # Acquire folder path for saving all model results and create folder
    MODEL_RESULTS_FOLDER_PATH = os.path.join(RESULTS_FOLDER, model.__class__.__name__, model.source,
                                             model.timestamp)
    create_folder(MODEL_RESULTS_FOLDER_PATH)

    model = model.to(device)

    print(model.__class__.__name__ + ': \n' + model.get_attributes_str())

    batch_loss_dict = {}
    batch_loss_dict['avg_over_n_batches'] = {}
    batch_loss_dict['raw'] = {}

    epoch_loss_dict = {}
    for loss_set in ['train', 'val']:
        epoch_loss_dict[loss_set] = {}
        for index_type in ['epoch_indexed', 'batch_indexed']:
            epoch_loss_dict[loss_set][index_type] = {}

    # Create training and validation datasets
    print(f'Model trained to isolate {model.source}')
    training_set = CACUnetInput('train', model.source, filter_train_val=filter_train)
    validation_set = CACUnetInput('val', model.source, filter_train_val=filter_val)

    # Print dataset sizes
    print(f'Training set has {len(training_set)} instances.')
    print(f'Validation set has {len(validation_set)} instances.')

    # Create training and validation data loaders
    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=batch_size,
                                                    num_workers=0)

    # Create checkpoint folder
    checkpoint_folder_path = os.path.join(MODEL_RESULTS_FOLDER_PATH, 'model_checkpoints')
    create_folder(checkpoint_folder_path)

    best_vloss = 100000000.

    # Define loss
    if loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss == 'L1':
        loss_fn = nn.L1Loss()
    else:
        print('Training failed, loss argument must be one of \n\'MSE\' - Mean Squared Error '
              '\n\'L1\' - Mean Absolute Error')
        return

    # Define optimiser
    default_lr = lr is None
    default_betas = betas is None
    default_weight_decay = weight_decay is None

    if default_lr:
        if default_betas:
            if default_weight_decay:
                optimizer = torch.optim.Adam(model.parameters())
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default parameters.')
                lr = 'Pytorch default'
                betas = 'Pytorch default'
                weight_decay = 'Pytorch default'
            else:
                optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default LR and betas.')
                lr = 'Pytorch default'
                betas = 'Pytorch default'
        else:
            if default_weight_decay:
                optimizer = torch.optim.Adam(model.parameters(), betas=betas)
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default LR and weight decay.')
                lr = 'Pytorch default'
                weight_decay = 'Pytorch default'
            else:
                optimizer = torch.optim.Adam(model.parameters(), betas=betas, weight_decay=weight_decay)
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default LR.')
                lr = 'Pytorch default'
    else:
        if default_betas:
            if default_weight_decay:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default betas and weight decay.')
                betas = 'Pytorch default'
                weight_decay = 'Pytorch default'
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default betas.')
                betas = 'Pytorch default'
        else:
            if default_weight_decay:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
                print(optimizer.__class__.__name__ + ' optimiser: Using pytorch default weight decay.')
                weight_decay = 'Pytorch default'
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
                print(optimizer.__class__.__name__ + ' optimiser: Using user-defined hyperparameters.')

    print(f'\n\nAll results relating to model will be saved to {MODEL_RESULTS_FOLDER_PATH}')

    training_start_time = time.time()

    for epoch in range(max_epochs):
        print(f'\n\n{model.__class__.__name__} ({model.source}): EPOCH {epoch + 1}:\n')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        epoch_avg_loss, batch_number = train_one_epoch(model, optimizer, loss_fn, training_loader, epoch,
                                                       batch_loss_dict, report_per_n_batches, device=device)
        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.
        for i, vdata in enumerate(validation_loader):
            vinput, vtarget = vdata
            vinput, vtarget = vinput.to(device), vtarget.to(device)
            voutput = model(vinput)
            if model.spec_output:
                # Convert to audio
                voutput = icomplex_as_channel(voutput)
                voutput = ispectro(voutput, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)
            # Calculate loss
            vloss = loss_fn(voutput, vtarget)
            running_vloss += vloss.item()

            # Delete validation loss and output to free up memory
            del vloss
            del voutput

        epoch_avg_vloss = running_vloss / (i + 1)  # Take average over all chunks in validation set
        print(f'Average batch loss over epoch || train: {epoch_avg_loss} valid: {epoch_avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        epoch_loss_dict['train']['epoch_indexed'][epoch + 1] = epoch_avg_loss
        epoch_loss_dict['val']['epoch_indexed'][epoch + 1] = epoch_avg_vloss
        epoch_loss_dict['train']['batch_indexed'][batch_number] = epoch_avg_loss
        epoch_loss_dict['val']['batch_indexed'][batch_number] = epoch_avg_vloss

        # Save the model's state
        checkpoint_file_path = os.path.join(checkpoint_folder_path, f'epoch_{epoch}')
        torch.save(model.state_dict(), checkpoint_file_path)
        # Track the best validation loss and corresponding epoch number
        if epoch_avg_vloss < best_vloss:
            best_vloss = epoch_avg_vloss
            best_vloss_epoch = epoch

        # Implement early stopping with specified patience
        if epoch - best_vloss_epoch > early_stopping_patience and epoch + 1 >= min_epochs:
            break

    # Acquire model training time
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    avg_epoch_time = training_time / (epoch + 1)

    print(f'Trained model saved to {checkpoint_file_path}')
    print(f'Training stopped after {epoch + 1} epochs')

    # Acquire path of checkpoint with the best validation loss and load model
    best_checkpoint_path = os.path.join(checkpoint_folder_path, f'epoch_{best_vloss_epoch}')
    print(f'Loading {model.timestamp} checkpoint from {best_checkpoint_path}')
    model.load_state_dict(torch.load(best_checkpoint_path))

    # Acquire model attributes string
    model_attributes_str = model.get_attributes_str() + f'\nEpoch index with best val loss: {best_vloss_epoch}'

    # Acquire training time string
    training_time_str = f'TRAINING TIME: \nModel total training time {training_time} seconds. \nModel average epoch ' \
                        f'time {avg_epoch_time} seconds.'
    print(training_time_str)

    # Acquire total number of model parameters
    num_params_str = model.get_num_params()
    # Acquire model size (MB)
    model_size_mb_str = get_model_size_mb(model)
    # Aquire model size summary string
    size_str = f'MODEL SIZE: \n{num_params_str} \n{model_size_mb_str}'
    print(size_str)

    # Acquire training hyperparameters string
    training_hyperparam_string = f'TRAINING HYPERPARAMETERS: \nBatch size: {batch_size}\nMaximum epochs: {max_epochs} ' \
                                 f'\nMinimum epochs: {min_epochs} \nEarly stopping patience: {early_stopping_patience}' \
                                 f'\nLoss function: {loss_fn.__class__.__name__} ' \
                                 f'\nOptimiser: {optimizer.__class__.__name__} \nLearning rate: {lr} ' \
                                 f'\nBetas: {betas} \nWeight decay: {weight_decay} \nFiltered train set: ' \
                                 f'{filter_train} \nFiltered validation set: {filter_val}'

    # Acquire model architecture string
    architecture_str = f'MODEL ARCHITECTURE: \n{repr(model)}'

    training_info_file_path = os.path.join(MODEL_RESULTS_FOLDER_PATH, model.timestamp + '_training_info.txt')

    with open(training_info_file_path, 'w') as f:
        f.write(model_attributes_str + '\n\n' + training_time_str + '\n\n' + size_str + '\n\n' +
                training_hyperparam_string + '\n\n' + architecture_str)
    print(f'Model attributes, size, training time and architecture as well as training hyperparameter details saved to '
          f'{training_info_file_path}')

    graph_folder_path = os.path.join(MODEL_RESULTS_FOLDER_PATH, 'graphs')
    create_folder(graph_folder_path)
    create_training_graphs(batch_loss_dict, epoch_loss_dict, graph_folder_path, model.timestamp)

    plot_chunk_specs(model, MODEL_RESULTS_FOLDER_PATH, device=device)

    if eval_after_training:
        if eval_on_test_set:
            eval_model(model, device=device)
        if eval_on_train_set:
            eval_model(model, subset='train', device=device)
        if eval_on_val_set:
            eval_model(model, subset='val', device=device)

    if apply_after_training:
        wav_file_name = 'Al James - Schoolboy Facination'
        _ = apply_model(model, wav_file_name, device=device)

    print(f'\n\nAll results relating to model have been saved to {MODEL_RESULTS_FOLDER_PATH}')

    del model


if __name__ == '__main__':
    if USE_CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training device: {device}')

    '''Creating single timestamp so both models share same file name in model checkpoints, runs, 
        training info and test results. This will make them easily identifiable when comparing models'''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create models
    source = 'vocals'  # Set source to be isolated
    model_list = []
    if TRAIN_STANDARD_TEST_MODEL:
        test_net = TestNet(num_layers=NUM_LAYERS_STANDARD, num_channels=NUM_CHANNELS_STANDARD, kernel_size=4, stride=2,
                           source='vocals', spec_output=False, timestamp=timestamp)
        model_list.append(test_net)
    if TRAIN_QUATERNION_TEST_MODEL:
        qtest_net = QTestNet(num_layers=NUM_LAYERS_STANDARD, num_channels=NUM_CHANNELS_STANDARD, kernel_size=4,
                             stride=2, source='vocals', spec_output=False, timestamp=timestamp)
        model_list.append(qtest_net)
    if TRAIN_STANDARD_MODEL:
        unet_model = UNet(num_channels=NUM_CHANNELS_STANDARD, num_layers=NUM_LAYERS_STANDARD, source=TYPE,
                          spec_output=SPEC_OUTPUT, timestamp=timestamp)
        model_list.append(unet_model)
    if TRAIN_QUATERNION_MODEL:
        qunet_model = QUNet(num_channels=NUM_CHANNELS_QUATERNION, num_layers=NUM_LAYERS_QUATERNION, source=TYPE,
                            spec_output=SPEC_OUTPUT, timestamp=timestamp)
        model_list.append(qunet_model)

    for model in model_list:
        train_model(model, LR, BETAS, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, REPORT_PER_N_BATCHES,
                    filter_train=FILTER_TRAIN_SET, filter_val=FILTER_VAL_SET, eval_after_training=EVAL_AFTER_TRAINING,
                    eval_on_test_set=EVAL_ON_TEST_SET, eval_on_train_set=EVAL_ON_TRAIN_SET,
                    eval_on_val_set=EVAL_ON_VAL_SET, apply_after_training=APPLY_AFTER_TRAINING, device=device)
