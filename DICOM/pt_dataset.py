# Wird kurz von dem Training in "Segmentation_Monai_PTLightning.ipynb" aufgerufen um die Daten (die PyTorch Files) richtig zu laden

import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """
    Loading the Datasets
    """

    def __init__(self, directory, augmentations=False):
        self.directory = directory # 1)
        self.augmentations = augmentations

        self.images = os.listdir(directory) # 2) Liste von allen Files in directory (alle Images)


    # def augment_gaussian_noise(self, data_sample, noise_variance=(0.001, 0.05)):
    #     # https://github.com/MIC-DKFZ/batchgenerators
    #     if noise_variance[0] == noise_variance[1]:
    #         variance = noise_variance[0]
    #     else:
    #         variance = random.uniform(noise_variance[0], noise_variance[1])
    #     data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    #     return data_sample

    def __len__(self):
        return len(os.listdir(self.directory)) # 3) Anzahl an Files in directory (Anzahl Bilder)

    def __getitem__(self, idx): # idx = Anzahl Aufrufe (Iteration 0,1,2,3,...)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image + lable
        name = self.images[idx] # 5) image: Liste von allen Files in directory (alle Images) [idx: geht Liste durch, eins Laden pro Aufruf]
        file = torch.load(os.path.join(self.directory, name)) # 6) Läd ein File pro Aufruf

        # Image / Lable trennen
        image = file["vol"]
        mask = file["mask"]

        return image, mask



# Nur zum Testen:

if __name__ == '__main__':

    # Pfad: Ordner mit einer PyTorch file (.pt) für jedes Bild
    path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Challenge_COVID-19-20_v2/Train_tensor_slices_filter"

    # Aufrufen mit meinem Data Loader
    dataset = TorchDataset(path, augmentations=False)

    # Visualisieren mit BatchViewer
    img, mask = dataset[1]
    img, mask = dataset[2]
    from batchviewer import view_batch # https://github.com/FabianIsensee/BatchViewer -> Console:  $ git clone https://github.com/FabianIsensee/BatchViewer.git $ cd BatchViewer $ pip install .
    view_batch(img, mask, width=512, height=512)