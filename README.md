# MedicalImages_To_PytorchFiles

## DICOM:

### "3DDICOM_to_2DTorch.py"

Imput: 
- DICOM Images 

Output: 
- PyTorch File (.pt) for each slice 

 At the bottom, there are the "ToDos" -> start there
 
 ### "3DDICOM_to_2Djpeg.py"

Imput: 
- DICOM Images 

Output: 
- jpeg file for each slice 

 At the bottom, there are the "ToDos" -> start there

### "3DDICOM_to_2DTorch_mask.py"

Imput: 
- DICOM Images of one Folder with several DICOM Files per 3d Image 
- DICOM Images with segmentation from syngo.via of one Folder with several DICOM Files per 3d Image (or with samll changes in the Programm also with Numpy Masks (follow ###))

Output: 
- PyTorch File (.pt) for each slice with the corresbonding mask

At the bottom, there are the "ToDos" -> start there 

### "Visualize_PytorchFiles.py"

Visualize the PyTorch images

### "Segmentation_Monai_PTLightning.ipynb"

Tutorial from Monai (https://monai.io/) with slightly changes, for Segmentation with PyTorch Lighning 

### "pt_dataset.py"

This is how the PyTorch data generated in "3DDICOM_to_2DTorch.py" can be loade druing training.
I call this in "Segmentation_Monai_PTLightning.ipynb" just before the training so that the data is loaded correctly.
This can be used for any other training setup as well 
