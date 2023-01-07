import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd

notebooks_path = Path.cwd()
lab_path = notebooks_path.parent
os.chdir(str(notebooks_path))

#Util

def save_as_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def open_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def show_slice(img, slice_num, cmap='gray'):
    plt.figure(figsize=(10,10))
    plt.imshow(img[slice_num], cmap=cmap)
    plt.show()
    
def my_argmax(array):
    """argnmax function but turns intensity cases where more than one tissue has a pick probability to -1"""
    rows = np.where(array == array.max(axis=1)[:, None])[0]
    rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]
    my_argmax = array.argmax(axis=1)
    my_argmax[rows_multiple_max] = -1
    return my_argmax

def dice_score(img1,img2):
    """Dice score

    Args:
        img1 (3D array): FIrst image to compute Dice from
        img2 (3D array): FIrst image to compute Dice from

    Returns:
        float: Dice score value
    """
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    dice = (2*np.sum(intersection))/(np.sum(union)+np.sum(intersection))
    return dice

# Read patients in the test-set:
def getPatientNum():
    """gets patients id number of the test set

    Returns:
        _type_: _description_
    """
    test_set_path = str(lab_path) + '/data/test-set'
    aux_patients_list = os.listdir(test_set_path + '/testing-images')
    patients_list = []
    for pat in aux_patients_list:
        # Remove ".nii.gz" in the "testing-images" folder
        patients_list.append(pat.replace(".nii.gz", ""))
    return sorted(list(map(int, patients_list))) #return sorted version of the patients list

def getArrayfromPath(image_path, dtype = np.uint16):
    """Simply get array of image

    Args:
        image_path (str): image path

    Returns:
        array: array of the image
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(dtype)

def get_testImage(id, skull_strip=False):
    """returns array of image given its path

    Args:
        id (int): id of patient (pat num)

    Returns:
        array: 3D image array
    """
    path_im = str(lab_path) + f'/data/test-set/testing-images/{id}.nii.gz'
    im = getArrayfromPath(path_im)
    if skull_strip:
        mask = getArrayfromPath(str(lab_path) + f'/data/test-set/testing-mask/{id}_1C.nii.gz')
        im_ns = im * mask
        return im_ns
    return im

def get_testLabels(id):
    path_im = str(lab_path) + f'/data/test-set/testing-labels/{id}_3C.nii.gz'
    im = getArrayfromPath(path_im)
    return im

def get_testMask(id):
    path_im = str(lab_path) + f'/data/test-set/testing-mask/{id}_1C.nii.gz'
    im = getArrayfromPath(path_im)
    return im

# save the array as a new nifti image
def save_as_nifti(array, filename, reference_path, dtype = np.uint16):
    """Save array as nifti image

    Args:
        array (array): array to be saved
        filename (str): path to save
        reference_path (str): path of reference image
        dtype (type, optional): data type of the array. Defaults to np.uint16.
    """
    reference_path = sitk.ReadImage(reference_path)
    image = sitk.GetImageFromArray(array.astype(dtype))
    image.SetOrigin(reference_path.GetOrigin())
    image.SetSpacing(reference_path.GetSpacing())
    image.SetDirection(reference_path.GetDirection())
    sitk.WriteImage(image, filename)
    
## Bias field removal
def bias_correct(inputImage):
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    return sitk.N4BiasFieldCorrection(inputImage)

# Intensity normalization
def intensity_normalize(inputImage):
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    return sitk.Normalize(inputImage)

# Denoising filter
def denoise(inputImage):
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    return sitk.DiscreteGaussian(inputImage)

def scale_to_bin_ratio(normalized_array, bin_ratio):
    """scales images using the bin_ratio so the output array has ine integer intensity values

    Args:
        normalized_array (array): image array normalised between 0 and 1
        bin_ratio (float): ration ebtween the maximum intensity of the original image and the maximum intensity of the preprocessed image

    Returns:
        _type_: _description_
    """
    return np.rint(normalized_array/bin_ratio).astype(np.uint16)

def sitk_like(array:np.ndarray, reference_sitk:sitk.Image):
    """takes an array and returns its sitk version copying the spacial info from the reference_sitk

    Args:
        array (np.ndarray): input array
        reference_sitk (sitk.Image): reference sitk image

    Returns:
        sirk.Image: output sitk image
    """
    array_to_sitk = sitk.GetImageFromArray(array)
    array_to_sitk.CopyInformation(reference_sitk)
    
    return array_to_sitk

def compute_hausdorff(groundT:np.ndarray, segmentation:np.ndarray, reference_sitk:sitk.Image):
    """compute hausdorff distance between ground truth and segmentation
    the inputs are arrays of the ground truth and segmentation and the reference_sitk is the sitk version of the ground truth

    Args:
        groundT (np.ndarray): gorund truth
        segmentation (np.ndarray): segmentation array
        reference_sitk (sitk.Image): refrence sitk image

    Returns:
        list: list of the hausdorff distances for each tissue
    """
    HD_distances= []
    for tissue in range(1,4):
        #first we convert the images back to sitk
        groundT_tissue = (groundT==tissue).astype(np.uint8)
        groundT_tissue = sitk_like(groundT_tissue, reference_sitk)
        segmentation_tissue = (segmentation==tissue).astype(np.uint8)
        segmentation_tissue = sitk_like(segmentation_tissue, reference_sitk)
        #define the HD filter
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        #compute distance
        hausdorff_distance_filter.Execute(groundT_tissue, segmentation_tissue)
        distance = hausdorff_distance_filter.GetHausdorffDistance()
        #append in list
        HD_distances.append(distance)
        
    return HD_distances