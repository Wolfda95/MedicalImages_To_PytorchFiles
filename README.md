# MedicalImages_To_PytorchFiles

## DICOM:

### "3DDICOM_to_2DTorch.py"

Imput: 
- DICOM Images of one Folder with several DICOM Files per 3d Image 
- DICOM Images with segmentation from syngo.via of one Folder with several DICOM Files per 3d Image (or with samll changes in the Programm also with Numpy Masks (follow ###))

Output: 
- PyTorch File (.pt) der das eine Schicht von einem Bild und die dazugehörige Schicht der Maske ernthält

Unten sind die "ToDos" -> da starten 

### "Visualize_PytorchFiles.py"

Damit kann man die PyTorch Bilder anschauen 

### "Segmentation_Monai_PTLightning.ipynb"

Tutorial von Monai (https://monai.io/) mit leichten Abänderungen von mit für Segmenterung mit PyTorch Lighning 

### "pt_dataset.py"

So können die in "3DDICOM_to_2DTorch.py" erzeugten PyTorch Daten geladen werden (das rufe ich in "Segmentation_Monai_PTLightning.ipynb" kurz vor dem Training auf) 
