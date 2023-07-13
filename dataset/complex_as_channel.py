import torch

from settings import NFFT, HOP_LENGTH, FRAMES_PER_SAMPLE

'''
spectro and ispectro functions within this file are taken from hybrid demucs code with minor adaptations:
repo url: https://github.com/facebookresearch/demucs
file url: https://github.com/facebookresearch/demucs/blob/main/demucs/spec.py

complex_as_channel and icomplex_as_channel are also taken from the hybrid demucs repo (class methods _mag and _mask):
file url: https://github.com/facebookresearch/demucs/blob/main/demucs/hdemucs.py

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
'''


def spectro(x, n_fft=NFFT, hop_length=HOP_LENGTH, pad=0):
    x = torch.from_numpy(x).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(x,
                   n_fft * (1 + pad),
                   hop_length or n_fft // 4,
                   window=torch.hann_window(n_fft).to(x),
                   win_length=n_fft,
                   normalized=True,
                   center=True,
                   return_complex=True,
                   pad_mode='reflect')
    del x
    z = z.to('cpu')
    _, freqs, frames = z.shape
    if freqs % 2 != 0:
        z = z[:, :-1, :]  # Remove final freq if num freqs is odd
        freqs -= 1
    if frames % 2 != 0:
        z = z[:, :, :-1]  # Remove final frame if num frames is odd
        frames -= 1
    return z.view(*other, freqs, frames)


def ispectro(z, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = torch.istft(z,
                    n_fft,
                    hop_length,
                    window=torch.hann_window(win_length).to(z.real),
                    win_length=win_length,
                    normalized=True,
                    length=length,
                    center=True)
    _, length = x.shape
    return x.view(*other, length)


def complex_as_channel(z):
    '''Combining audio channel and complex dimensions for complex-as-channel representation'''
    if z.dim() == 3:
        channels, freqs, frames = z.shape
        z = torch.view_as_real(z).permute(0, 3, 1, 2)
        return z.reshape(channels * 2, freqs, frames)
    elif z.dim() == 4:
        sources, channels, freqs, frames = z.shape
        z = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        return z.reshape(sources, channels * 2, freqs, frames)
    else:
        print('Input for complex as channel should have 3 or 4 dimensions.')
        return


def icomplex_as_channel(z):
    '''Split CAC dimension into audio channel and complex dimensions to prepare for iSTFT'''
    instances, channels, freqs, frames = z.shape
    out = z.view(instances, -1, 2, freqs, frames).permute(0, 1, 3, 4, 2)
    out = torch.view_as_complex(out.contiguous())
    return out
