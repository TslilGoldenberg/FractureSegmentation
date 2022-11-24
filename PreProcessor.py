import numpy as np
import skimage
from skimage.measure import label
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
import nibabel as nib
import cv2 as cv2
import dcm2niix
import os
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist, equalize_hist
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import dicom2nifti

MAX_VOXEL_VALUE = 65535
MIN_VOXEL_VALUE = 0
FAILURE = -1
DERIVATIVE_KERNEL = [1, -1]


class PreProcessor:
    def __init__(self, nifti_file: str):
        self.nifty_file = nifti_file
        self.raw_img = nib.load(nifti_file)
        self.img_data = self.raw_img.get_fdata().astype(dtype=np.uint16)
        _, _, z_slices = np.nonzero(self.img_data)
        self.bottom_bound_slice, self.top_bound_slice = np.min(z_slices), np.max(z_slices)
        self.equalized_img = None
        self.bones = None

    def _applyThreshold(self, minTH, maxTH):
        """

        :param minTH:
        :param maxTH:
        :return:
        """
        pass

    def SegmentationByTH(self, Imin, Imax):
        """
        This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
        The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
        :param nifty_file:
        :param Imin:
        :param Imax:
        :return:
        """
        if self.equalized_img is None:
            self.equalized_img = np.copy(self.img_data)
            self.equalized_img = equalize_adapthist(self.equalized_img, nbins=MAX_VOXEL_VALUE) * (MAX_VOXEL_VALUE)
            self.equalized_img = self.equalized_img.astype(dtype=np.uint16)


        self.equalized_img[(self.equalized_img <= Imax) & (self.equalized_img > Imin)] = MAX_VOXEL_VALUE
        self.equalized_img[self.equalized_img < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
        opened_img = skimage.morphology.closing(self.equalized_img)
        return opened_img


    def SkeletonTHFinder(self):
        """
        This function iterates over 25 candidate Imin thresholds in the range of [150,500] (with intervals of 14).
        In each run, use the SegmentationByTH function you’ve implemented, and count the number of connectivity components
        in the resulting segmentation with the current Imin. Plot your results – number of connectivity components per Imin. Choose the Imin which is the first or second minima in the plot. Also, make sure to include that graph in your report.

        Next, you need to perform post-processing (morphological operations – clean out single pixels, close holes, etc.)
        until you are left with a single connectivity component.
        Finally, this function should save a segmentation NIFTI file called “<nifty_file>_SkeletonSegmentation.nii.gz” and
        return the Imin used for that.
        :return:
        """
        Imin_range = np.arange(150, 300, 10)
        best_result = np.inf
        img_res = None
        for i_min in Imin_range:
            img = self.SegmentationByTH(i_min, 1300)
            labels, cmp = label(self.img_data, return_num=True)
            if cmp == 0:
                break
            if best_result > cmp:
                best_result = cmp
                img_res = img
        return img_res


    def _extractPelvic(self):
        # Step 1: Apply Adaptive Histogram Equalizer

        # Step 2: Apply Thresholding for Skeleton
        self.bones = self.SkeletonTHFinder()
        final_image = nib.Nifti1Image(self.bones, self.raw_img.affine)
        nib.save(final_image, "out_seg.nii.gz")
        # Step 3: Get a general ROI

    def get_Nlargest_component(self, output_directory=""):
        labels = label(self.bones)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        largestCC_img = self.bones*largestCC
        largestCC_img = skimage.morphology.closing(largestCC_img)
        final_image = nib.Nifti1Image(largestCC_img, self.raw_img.affine)
        nib.save(final_image, f"{output_directory}out_largestCC.nii.gz")

    def get_pelvis_ROI(self):
        bin_res = np.zeros(self.img_data.copy().shape)
        bin_res[self.img_data > 0] = 1
        slices = np.sum(bin_res, axis=(0,1))
        plt.plot(np.arange(slices.shape[0]), slices)
        plt.show()
        res = bin_res[:,:,100:400]
        final_image = nib.Nifti1Image(res, self.raw_img.affine)
        nib.save(final_image, "ROI_test.nii.gz")

    @staticmethod
    def convert_files_from_dicom2nifti(num_files: int):
        for i in range(1, num_files+1):
            dicom_directory = f"dicom_directory/{i}"
            output_directory = f"output_directory_{i}"
            os.mkdir(output_directory)
            dicom2nifti.convert_directory(dicom_directory, output_directory, compression=True, reorient=True)


if __name__ == "__main__":
    for i in range(1, 2):
        print(f"working on {i}")
        output_directory=f"output_directory_{i}/"
        nifty_file = f'{output_directory}0_anonymous.nii.gz'
        ob = PreProcessor(nifti_file=nifty_file)
        ob._extractPelvic()
        ob.get_Nlargest_component(output_directory=output_directory)

