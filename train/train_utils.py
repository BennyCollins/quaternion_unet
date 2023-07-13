import os

import matplotlib.pyplot as plt

from settings import REPORT_PER_N_BATCHES


def get_model_size_mb(model):
    param_bytes = 0
    buffer_bytes = 0
    # Acquire number of bytes for parameters
    for param in model.parameters():
        param_bytes += param.numel() * param.element_size()
    # Acquire number of bytes for buffers
    for buffer in model.buffers():
        buffer_bytes += buffer.numel() * buffer.element_size()
    return f'Model size: {(param_bytes + buffer_bytes) / 1024**2}MB'


def create_training_graphs(batch_loss_dict, epoch_loss_dict, graph_directory, model_id):
    # Build batch loss graphs
    plt.figure(1)
    plt.plot(batch_loss_dict['raw'].keys(), batch_loss_dict['raw'].values(), label='Train batch loss', linestyle="-",
             color='seagreen', zorder=0)
    plt.plot(batch_loss_dict['avg_over_n_batches'].keys(), batch_loss_dict['avg_over_n_batches'].values(),
             label=f'Train loss (mean over {REPORT_PER_N_BATCHES} batches)', linestyle="-", linewidth=0.75,
             color='tomato', zorder=5)
    plt.scatter(epoch_loss_dict['val']['batch_indexed'].keys(), epoch_loss_dict['val']['batch_indexed'].values(),
             label='Validation loss', color='orange', zorder=15)
    plt.scatter(epoch_loss_dict['train']['batch_indexed'].keys(), epoch_loss_dict['train']['batch_indexed'].values(),
                label='Train loss (mean over batches in epoch)', color='cornflowerblue', zorder=10)

    plt.title('Training and validation loss')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(graph_directory, model_id + '_loss_over_batches.jpg'))
    plt.clf()

    plt.figure(2)
    plt.plot(batch_loss_dict['raw'].keys(), batch_loss_dict['raw'].values(), label='Train batch loss', linestyle="-",
             color='seagreen', zorder=0)
    plt.plot(batch_loss_dict['avg_over_n_batches'].keys(), batch_loss_dict['avg_over_n_batches'].values(),
             label=f'Train loss (mean over {REPORT_PER_N_BATCHES} batches)', linestyle="-", linewidth=0.75,
             color='tomato', zorder=5)

    plt.title('Training loss')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(graph_directory, model_id + '_train_loss_over_batches.jpg'))
    plt.clf()

    # Build epoch loss graphs
    plt.figure(3)
    plt.plot(epoch_loss_dict['train']['epoch_indexed'].keys(), epoch_loss_dict['train']['epoch_indexed'].values(),
             label='Train loss (calculated during epoch)', linestyle="-", color='seagreen')
    plt.plot(epoch_loss_dict['val']['epoch_indexed'].keys(), epoch_loss_dict['val']['epoch_indexed'].values(),
             label='Validation loss', linestyle="-", color='tomato')
    plt.title('Training and validation loss \nper epoch')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(graph_directory, model_id + '_loss_over_epochs.jpg'))
    plt.clf()