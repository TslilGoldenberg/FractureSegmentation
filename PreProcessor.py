import numpy as np
import skimage
from skimage.measure import label
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
import nibabel as nib
from matplotlib import pyplot as plt
import multiprocessing

from skimage.exposure import equalize_adapthist, equalize_hist
import pydicom as dcm
import threading

from glob import glob
import dcmstack
import pandas as pd
import os
from torchviz import make_dot
import dcm2niix
import dicom2nifti

MAX_VOXEL_VALUE = 65535
MIN_VOXEL_VALUE = 0
FAILURE = -1
DERIVATIVE_KERNEL = [1, -1]
NIFTI = 'nii'

class PreProcessor:
    def __init__(self, file_path: str, output_directory: str, resolution=10):
        self.resolution = resolution
        self._dips = []
        self.isNIFTI = True if file_path.split('.').__contains__(NIFTI) else False
        self.img_header = {}
        self.raw_img = None
        self.output_directory = output_directory
        self._read_file(file_path)
        self.img_data = self.getImageData()
        _, _, z_slices = np.nonzero(self.img_data)
        self.bottom_bound_slice, self.top_bound_slice = np.min(z_slices), np.max(z_slices)
        self.equalized_img = None
        self.bones = None




    def _read_file(self, file_path:str):
        if self.isNIFTI:
            self.raw_img = nib.load(file_path)
            return
        file = file_path.split('/')[:-1]
        src_dcms = glob(f"{'/'.join(file)}/*.dcm")

        stacks = dcmstack.parse_and_stack(src_dcms)
        nifti = None
        for stack in stacks.values():
            nifti = stack.to_nifti(embed_meta=True)
        self.raw_img = nifti


    def transform_to_hu(self):
        self.img_data *= float(self.img_header['slope'])
        self.img_data += float(self.img_header['inter'])


    def SegmentationByTH(self, Imin, Imax):
        """
        This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
        The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
        :param nifty_file:
        :param Imin:
        :param Imax:
        :return:
        """

        img = np.copy(self.img_data.astype(dtype=np.uint16))
        # img = np.array(equalize_adapthist(img/MAX_VOXEL_VALUE, nbins=MAX_VOXEL_VALUE) * MAX_VOXEL_VALUE, dtype=np.uint16).clip(0,MAX_VOXEL_VALUE)

        img[(img <= Imax) & (img > Imin)] = MAX_VOXEL_VALUE
        img[img < MAX_VOXEL_VALUE] = 0
        opened_img = skimage.morphology.opening(img)
        closed_img = skimage.morphology.closing(opened_img)

        return closed_img


    def SkeletonTHFinder(self):
        """
        This function iterates over 25 candidate Imin thresholds in the range of [150,500] (with intervals of 14).
        In each run, use the SegmentationByTH function you’ve implemented, and count the number of connectivity components
        in the resulting segmentation with the current Imin. Plot your results – number of connectivity components per Imin.
        Choose the Imin which is the first or second minima in the plot. Also, make sure to include that graph in your report.

        Next, you need to perform post-processing (morphological operations – clean out single pixels, close holes, etc.)
        until you are left with a single connectivity component.
        Finally, this function should save a segmentation NIFTI file called “<nifty_file>_SkeletonSegmentation.nii.gz” and
        return the Imin used for that.
        :return:
        """
        Imin_range = np.arange(150, 514, self.resolution)
        img_res = []
        ccmps = []

        for i_min in Imin_range:
            img = self.SegmentationByTH(i_min, 1500)
            _, cmp = label(img, return_num=True)
            print(f"cmp:{cmp} and imin: {i_min}")
            ccmps.append(cmp)
            img_res.append(img)
        self.find_first_minima(ccmps)
        self._get_intensities_hist()
        self.bones = img_res[self._dips[1]]
        return self.bones

    def _get_intensities_hist(self):
        _, _, patches = plt.hist(self.img_data.flatten().astype(dtype=np.uint16), bins=MAX_VOXEL_VALUE)
        plt.ylim(1, 40000)
        plt.xlim(100, 1500)
        d = []
        for opt_thresh in self._dips:
            opt_thresh *= 10
            opt_thresh += 150
            d.append(opt_thresh)
            for p in patches[opt_thresh: opt_thresh+3]:
                p.set_fc("red")
            for p in patches[opt_thresh-3:opt_thresh]:
                p.set_fc("red")

        plt.title(f"Intensities histogram. dips in {d}")
        plt.savefig(self.output_directory + "histogram")
        plt.show()

    def find_first_minima(self,connectivity_cmps):
        """

        :return:
        """
        minimas = np.array(connectivity_cmps)
        # Finds all local minima
        self._dips = np.where((minimas[1:-1] < minimas[0:-2]) * (
                minimas[1:-1] < minimas[2:]))[0]


    def extract_pelvis_bone(self):
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
        largestCC_img = skimage.morphology.opening(largestCC_img)
        self.get_pelvis_ROI()
        final_image = nib.Nifti1Image(largestCC_img, self.raw_img.affine)
        nib.save(final_image, f"{output_directory}out_largestCC.nii.gz")

    def get_pelvis_ROI(self):
        bin_res = np.zeros(self.img_data.copy().shape)
        bin_res[self.bones > 0] = 1
        slices = np.sum(bin_res, axis=(0,1))
        plt.plot(np.arange(slices.shape[0]), slices)
        plt.title(f"Density level in CT with {self.img_data.shape[2]} slices")
        plt.xlabel("Slices index")
        plt.ylabel("Intensity Density")
        plt.savefig(f"{self.output_directory}density")
        plt.show()
        # res = bin_res[:,:,100:400]
        # final_image = nib.Nifti1Image(bin_res, self.raw_img.affine)
        # nib.save(final_image, self.output_directory+"ROI_test.nii.gz")

    def getImageData(self):
        return self.raw_img.get_fdata()

if __name__ == "__main__":
    for i in range (1,20):
        output_directory= f"output_directory_{i}/"
        file = f"dicom_directory/{i}/IM_0000.dcm"
        ob = PreProcessor(file_path=file, output_directory=output_directory)
        ob.extract_pelvis_bone()
        ob.get_Nlargest_component(output_directory=output_directory)

