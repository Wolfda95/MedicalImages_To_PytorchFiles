# 3D DICOM to 2D jpeg File für LIDC Daten

import glob # Pade einlesen
import pydicom as dicom # Dicom Einlesen
import cv2       # Numpy to jepg/png
from tqdm import tqdm
import scipy.ndimage # rezize: zoom

import os
import numpy as np
import torch

# ============================== Image ===========================================

# ------------------- DICOM Image to Numpy Array (+Houndfield) ------------------------------
def get_pixels_hu(scans):  # DICOM to Pixel

    image = scans.pixel_array  # CT Scan
    image = image.astype(np.int16) # to Numpy

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept if 'RescaleIntercept' in scans else -1024
    slope = scans.RescaleSlope if 'RescaleSlope' in scans else 1

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

# image normalization (for png/jepg)
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


# =============================================================================
# Save
# =============================================================================

def save(image, save_path, patient_name, i, w1, w2):

    # --------------------------Image----------------------------------------------------
    # DICOM -> Numpy
    patient_dicom = dicom.dcmread(image) # Dicom Datei einlesen

    patient_pixels = get_pixels_hu(patient_dicom)  # Numpy Array (x,y)
    patient_pixels = win_scale(patient_pixels, w1, w2, type(patient_pixels), [patient_pixels.min(), patient_pixels.max()])  # Numpy Array Korrigiert

    # Resize [48,800,800]
    #patient_pixels = scipy.ndimage.zoom(patient_pixels, (min(1, (48 / patient_pixels.shape[0])), (800 / patient_pixels.shape[1]), (800 / patient_pixels.shape[2])),mode="nearest", grid_mode=True)

    # Normalization for jpeg/png
    patient_pixels = interval_mapping(patient_pixels, patient_pixels.min(), patient_pixels.max(), 0, 255)
    patient_pixels = patient_pixels.astype(np.uint8)
    patient_pixels.astype(np.uint8)

    # Save
    path = save_path + "/" + str(patient_name) + "_" + str(i) + ".jpeg"
    cv2.imwrite(path, patient_pixels)


# =============================================================================
# Main
# =============================================================================
def main():

    # Daten: Images: DICOM Files in einem Ordner


    # ToDo: Pfade wo die Daten gespeichert sind:
    data_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/LIDC/manifest-1600709154662/LIDC-IDRI"

    # ToDo: Pfad wo die PyThorch Files gespeichert werden sollen
    save_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/LIDC/manifest-1600709154662/LIDC-2D-jpeg" # Mus vorher angelegt werden!!!!!!!

    # ToDo: Korrektur bei CT:
    wl = -400
    ww = 1500
    #body_part = "abdomen": wl = 60, ww = 400
    #body_part == "angio": wl = 300, ww = 600
    #body_part == "bone": wl = 300, ww = 150
    #body_part == "brain": wl = 40, ww = 80
    #body_part == "chest": wl = 40, ww = 400
    #body_part == "lungs": wl = -400, ww = 1500


    i = 0

    # ToDo: Je nach Ordnerstruktur anpassen:
    Ordner = sorted(glob.glob(data_path + "/*"))  # Liste: Alle Pfade aus dem Ordner LIDC-IDRI (Ordner der einzelnen Patienten)
    for fileA in Ordner:  # durchläuft alle Pfade im Ordner LIDC-IDRI (alle Patienten)
        patient_name = fileA.split("/")[-1]  # Patientenname

        Ordner2 = sorted(glob.glob(fileA + "/*")) # Liste: Alle Pfade aus einem Patientenordner (einmal Röntgen, einmal CT)
        for fileB in Ordner2: # durchläuft alle Pfade im Patientenordner (Röntgen und CT)

            Ordner3 = sorted(glob.glob(fileB + "/*"))  # unnötiger Unterordner
            for fileC in Ordner3:

                Ordner4 = sorted(glob.glob(fileC + "/*"))  # Pfade aller DICOm Files (entweder CT oder Röntgen)
                # Check ob CT oder Röntgen:
                number_files = len(Ordner4) # Anzahl der Files im Ordner
                if number_files > 10: # nur wenn mehr als 10 Files im Ordner == CT (da Röntgen nur wenige Files)
                    for fileD in Ordner4: # durchläuft alle DICOM Files

                        # Check ob dicom oder xml
                        if fileD.endswith(".dcm"):

                            # (Pfad der Serie (DICOM Files), Patientenname, Schicht Nummer, WL, WW)
                            save(fileD, save_path, patient_name, i, wl, ww)

                            i = i+1


if __name__ == '__main__':
    main()