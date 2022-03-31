# 3D DICOM to 2D PyTorch File with mask

import glob # Pade einlesen
import pydicom as dicom # Dicom Einlesen
from tqdm import tqdm
import scipy.ndimage # rezize: zoom

import matplotlib.pyplot as plt
from batchviewer import view_batch  # aus Git runtergeladen

import os
import numpy as np
import torch

# ============================== Image ===========================================

# -------------------Load DICOM Image (sort the files)------------------------------
def load_scan(path):
    slices = [dicom.dcmread(path + '/' + s) for s in sorted(os.listdir(path))] #holt alle DICOM Dateien aus dem Ordner
    # slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key=lambda x: int(x.InstanceNumber)) #InstanceNumber sagt an welcher Stelle die DICOM Datei kommen muss
    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
    #                              slices[1].ImagePositionPatient[2])
    # except:
    #
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    #
    # for s in slices:
    #     s.SliceThickness = slice_thickness
    return slices


# ------------------- DICOM Image to Numpy Array (+Houndfield) ------------------------------
def get_pixels_hu(scans):  # DICOM to Pixel

    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept if 'RescaleIntercept' in scans[0] else -1024
    slope = scans[0].RescaleSlope if 'RescaleSlope' in scans[0] else 1

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

# ------------------- CT: Scale pixel intensity --------------------------------------------
# von https://gist.github.com/lebedov/e81bd36f66ea1ab60a1ce890b07a6229
# abdomen: {'wl': 60, 'ww': 400} || angio: {'wl': 300, 'ww': 600} || bone: {'wl': 300, 'ww': 1500} || brain: {'wl': 40, 'ww': 80} || chest: {'wl': 40, 'ww': 400} || lungs: {'wl': -400, 'ww': 1500}
def win_scale(data, wl, ww, dtype, out_range):
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (
                out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)

# Normalisieren für PyTorch Arrays
def min_max_normalization(data, eps):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range

    return data_normalized


# ============================== Mask ===========================================

def load_scan_mask(path):
    # https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [dicom.dcmread(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices

# Segmentierung -> Mask
def seg_mask(path):
    slices = load_scan_mask(path)
    # create empty mask
    image = np.stack([s.pixel_array for s in slices])
    mask = np.zeros((image.shape[0:3]))

    """ in the images that are exported via the syngio.via tool all of the pixels that are not mask have the same values for RGB."""
    for i in tqdm(range(image.shape[0])):
        image_tmp = image[i, ...]

        # selecting the channels
        r_channel = image_tmp[..., 0]
        g_channel = image_tmp[..., 1]
        b_channel = image_tmp[..., 2]

        # iterating over x dimension.
        for x in range(image.shape[1]):
            # iterating over y dimension.
            for y in range(image.shape[2]):

                # comparision of the pixels have the same value.
                if r_channel[x, y] == g_channel[x, y] == b_channel[x, y]:

                    mask[i, x, y] = 0
                else:
                    mask[i, x, y] = 1

    # conversion into unsigned int 8 and saving in a pt file.
    mask = mask.astype(np.uint8)
    return mask


# =============================================================================
# Save
# =============================================================================

def save(mask, image, save_path, i, w1, w2):


    # --------------------------Image----------------------------------------------------
    # DICOM -> Numpy
    patient_dicom = load_scan(image) # Dicom Datei einlesen und Schichten richtig ordnen
    patient_pixels = get_pixels_hu(patient_dicom)  # Numpy Array (Anzahl Schichten, x,y)
    patient_pixels = win_scale(patient_pixels, w1, w2, type(patient_pixels), [patient_pixels.min(), patient_pixels.max()])  # Numpy Array Korrigiert
    #patient_pixels = patient_pixels[::-1,...]  # läuft die Schichten von hinten durch, da irgendwie die Schichten umgedreht wurden

    # convert to float32
    patient_pixels = patient_pixels.astype(np.float32)

    #print(patient_pixels.shape)

    #  Resize [48,800,800]
    #patient_pixels = scipy.ndimage.zoom(patient_pixels, (min(1, (48 / patient_pixels.shape[0])), (800 / patient_pixels.shape[1]), (800 / patient_pixels.shape[2])),mode="nearest", grid_mode=True)

    # Torch tensoren können keine negativen Zahlen -> deshalb normalisieren
    patient_pixels = min_max_normalization(patient_pixels, 0.001)

    # Resize [48,800,800]
    # patient_pixels = scipy.ndimage.zoom(patient_pixels, (min(1, (48 / patient_pixels.shape[0])), (800 / patient_pixels.shape[1]), (800 / patient_pixels.shape[2])),mode="nearest", grid_mode=True)

    # Numpy -> Torch
    pixel_image = torch.from_numpy(patient_pixels)
    pixel_image = pixel_image.to(torch.float16)

    # --------------------------Mask----------------------------------------------------

    # Maske erzeugen
    pixel_mask = seg_mask(mask) ### wenn schon Numpy, das weglassen

    # convert to float32
    pixel_mask = pixel_mask.astype(np.float32)

    # Resize [48,800,800]
    # pixel_mask = scipy.ndimage.zoom(pixel_mask, (min(1, (48 / pixel_mask.shape[0])), (800 / pixel_mask.shape[1]), (800 / pixel_mask.shape[2])), mode="nearest", grid_mode=True)

    # Numpy -> Torch
    pixel_mask = torch.from_numpy(pixel_mask)
    pixel_mask = pixel_mask.to(torch.float16)

    # ----------------------- 2D --------------------------------------------------------------------
    for i in range(len(pixel_image)):

        # jede 2d Schicht
        pixel_image = pixel_image[i]  # [x,y]
        pixel_mask = pixel_mask[i]

        # Eine Dim mehr für das Training [1,x,y]
        pixel_image = pixel_image.unsqueeze(0).float()
        pixel_mask = pixel_mask.unsqueeze(0).float()

        # Save
        path = save_path + "/" + str(i) + ".pt"
        torch.save({"vol": pixel_image, "mask": pixel_mask}, path)


# =============================================================================
# Main
# =============================================================================
def main():

    # Daten: Images: DICOM Files in einem Ordner || Segmentierung: DICOM Files mit syngo.via Segmentierung in einem Ordner
    # Falls Maske schon als Numpy: ### befolgen

    # ToDo: Pfade wo die Daten gespeichert sind:
    data_path = "/home/wolfda/Clinic_Data/Data/Leber/"

    # ToDo: Pfad wo die PyThorch Files gespeichert werden sollen
    save_path = "/home/wolfda/Clinic_Data/Data/Leber/PyTorch_files" # Mus vorher angelegt werden!!!!!!!

    # ToDo: Korrektur bei CT:
    wl = 60
    ww = 400
    #body_part = "abdomen": wl = 60, ww = 400
    #body_part == "angio": wl = 300, ww = 600
    #body_part == "bone": wl = 300, ww = 150
    #body_part == "brain": wl = 40, ww = 80
    #body_part == "chest": wl = 40, ww = 400
    #body_part == "lungs": wl = -400, ww = 1500


    i = 0

    # ToDo: Je nach Ordnerstruktur anpassen:
    Ordner = sorted(glob.glob(data_path + "*"))  # Liste: Alle Pfade aus dem Ordner Leber (Ordner der einzelnen Patienten)
    for fileA in Ordner:  # durchläuft alle Pfade im Ordner Leber (alle Patienten)
        Ordner2 = sorted(glob.glob(fileA + "/*")) # Liste: Alle Pfade aus einem Patientenordner (alle DICOM Files (=alle Schichten) enes Patienten)
        for fileB in Ordner2: # durchläuft alle Pfade im Patientenordner (alle DICOM Files)

            if fileB == glob.glob(fileB + "/000*")[0]: # Wenn es mit 000 anfängt ist es das Bild (weil Bild ist mit SAP ID gespeichert)
                image = fileB

            else: # Wenn es mit mit was anderem anfängt ist es die Maske (weil es ist mit der Seriennummer gespeichert)
                mask = fileB

            # (Pfad der Serie (DICOM Files), Pfad der Makse (DICOM Files mit syngo.via Segmentierung) ### alternativ hier den Pad zum Numpy File, Pfad wo die PyTorch Files gespeichert werden sollen, Durchnummerriert, CT Werte)
            save(mask, image, save_path, i, wl, ww)


if __name__ == '__main__':
    main()