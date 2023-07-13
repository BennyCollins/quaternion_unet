import os


'''
Function taken from the url below:
repo url: https://github.com/vskadandale/multichannel-unet-bss
file url: https://github.com/vskadandale/multichannel-unet-bss/blob/master/utils/utils.py

@inproceedings{kadandale2020multi,
  title={Multi-channel U-Net for Music Source Separation},
  author={Kadandale, Venkatesh S and Montesinos, Juan F and Haro, Gloria and G{\'o}mez, Emilia},
  booktitle={2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
'''


def create_folder(path):
    if not os.path.exists(path):
        os.umask(0)  # To mask the permission restrictions on new files/directories being create
        os.makedirs(path, 0o755)  # setting permissions for the folder
