from datetime import datetime

import click
import torch

from models.UNet import UNet
from models.QUNet import QUNet
from train.train import train_model
from settings import BETAS, WEIGHT_DECAY, SPEC_OUTPUT, REPORT_PER_N_BATCHES, USE_CUDA


@click.command()
@click.option('--source', default='vocals',
              help='Source to separate from mixture: vocals, accompaniment, drums, bass or other.')
@click.option('--batch_size', default=8, help='Batch size for training.')
@click.option('--lr', default=0.001, help='Learning rate for training.')
@click.option('--max_epochs', default=100, help='Maximum number of epochs for training.')
@click.option('--min_epochs', default=30, help='Minimum number of epochs for training.')
@click.option('--early_stopping_patience', default=10, help='Number of epochs to wait for validation error to improve '
                                                            'before stopping training.')
@click.option('--train_standard_unet', default=True, help='Train standard UNet.')
@click.option('--num_channels_standard', default=64, help='Number of channels after first conv of standard UNet.')
@click.option('--num_layers_standard', default=5, help='Number of hierarchical layers in standard UNet.')
@click.option('--num_dropout_enc_layers_standard', default=0,
              help='Number of encoder layers with dropout in standard UNet.')
@click.option('--num_dropout_dec_layers_standard', default=None,
              help='Number of decoder layers with dropout in standard UNet.')
@click.option('--dropout_standard', default=0.5, help='Degree of dropout in standard UNet.')
@click.option('--leakiness_standard', default=0.2,
              help='Degree of leakiness in standard UNet\'s leaky ReLU activations.')
@click.option('--train_quaternion_unet', default=True, help='Train quaternion UNet.')
@click.option('--num_channels_quaternion', default=64, help='Number of channels after first conv of quaternion UNet.')
@click.option('--num_layers_quaternion', default=5, help='Number of hierarchical layers in quaternion UNet.')
@click.option('--num_dropout_enc_layers_quaternion', default=0,
              help='Number of encoder layers with dropout in quaternion UNet.')
@click.option('--num_dropout_dec_layers_quaternion', default=None,
              help='Number of decoder layers with dropout in quaternion UNet.')
@click.option('--dropout_quaternion', default=0.5, help='Degree of dropout in quaternion UNet.')
@click.option('--leakiness_quaternion', default=0.2,
              help='Degree of leakiness in quaternion UNet\'s leaky ReLU activations.')
@click.option('--quaternion_dropout', default=True, help='Use quaternion dropout in QUNet.')
@click.option('--quaternion_norm', default=True, help='Use quaternion normalisation in QUNet.')
@click.option('--loss', default='MSE', help='Loss function - either \'MSE\' or \'L1\'.')
@click.option('--filter_train_set', default=False, help='Filter out training chunks with silent target audio.')
@click.option('--filter_val_set', default=False, help='Filter out validation chunks with silent target audio.')
@click.option('--eval_after_training', default=True, help='Evaluate the model after training.')
@click.option('--eval_on_train_set', default=True, help='Evaluate the model on the training set.')
@click.option('--eval_on_val_set', default=True, help='Evaluate the model on the validation set.')
@click.option('--apply_after_training', default=True, help='Apply the model to example wav after training.')
def main(source, batch_size, lr, max_epochs, min_epochs, early_stopping_patience, train_standard_unet, num_channels_standard,
         num_layers_standard, num_dropout_enc_layers_standard, num_dropout_dec_layers_standard, dropout_standard,
         leakiness_standard, train_quaternion_unet, num_channels_quaternion, num_layers_quaternion,
         num_dropout_enc_layers_quaternion, num_dropout_dec_layers_quaternion, dropout_quaternion, leakiness_quaternion,
         quaternion_dropout, quaternion_norm, loss, filter_train_set, filter_val_set, eval_after_training,
         eval_on_train_set, eval_on_val_set, apply_after_training):
    if USE_CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Training device: {device}')

    '''Creating single timestamp so both models share same file name in model checkpoints, runs, 
        training info and test results. This will make them easily identifiable when comparing models'''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create models
    model_list = []

    if train_standard_unet:
        unet_model = UNet(source=source, num_channels=num_channels_standard, num_layers=num_layers_standard,
                          num_dropout_enc_layers=num_dropout_enc_layers_standard,
                          num_dropout_dec_layers=num_dropout_dec_layers_standard, dropout=dropout_standard,
                          leakiness=leakiness_standard, spec_output=SPEC_OUTPUT, timestamp=timestamp)
        model_list.append(unet_model)

    if train_quaternion_unet:
        qunet_model = QUNet(source=source, num_channels=num_channels_quaternion, num_layers=num_layers_quaternion,
                            num_dropout_enc_layers=num_dropout_enc_layers_quaternion,
                            num_dropout_dec_layers=num_dropout_dec_layers_quaternion, dropout=dropout_quaternion,
                            quaternion_dropout=quaternion_dropout, quaternion_norm=quaternion_norm,
                            leakiness=leakiness_quaternion, spec_output=SPEC_OUTPUT, timestamp=timestamp)
        model_list.append(qunet_model)

    for model in model_list:
        train_model(model=model, lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY, max_epochs=max_epochs,
                    min_epochs=min_epochs, early_stopping_patience=early_stopping_patience, batch_size=batch_size,
                    loss=loss, report_per_n_batches=REPORT_PER_N_BATCHES, filter_train=filter_train_set,
                    filter_val=filter_val_set, eval_after_training=eval_after_training, eval_on_test_set=True,
                    eval_on_train_set=eval_on_train_set, eval_on_val_set=eval_on_val_set,
                    apply_after_training=apply_after_training, device=device)


if __name__ == '__main__':
    main()
