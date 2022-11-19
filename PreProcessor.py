import numpy as np
import skimage
from skimage.measure import label
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
import nibabel as nib
import cv2 as cv2
import dicom2nifti
import os
from matplotlib import pyplot as plt

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
        self.bones = None

    def _applyThreshold(self, minTH, maxTH):
        """

        :param minTH:
        :param maxTH:
        :return:
        """
        pass

    def SegmentationByTH(self, nifty_file, Imin, Imax):
        """
        This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
        The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
        :param nifty_file:
        :param Imin:
        :param Imax:
        :return:
        """
        img = nib.load(nifty_file)
        img_data = img.get_fdata().astype(dtype=np.uint16)
        img_data[(img_data <= Imax) & (img_data > Imin)] = MAX_VOXEL_VALUE
        img_data[img_data < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
        opened_img = skimage.morphology.closing(img_data)
        final_image = nib.Nifti1Image(opened_img, img.affine)
        nib.save(final_image, f"out_seg_{Imin}_{Imax}.nii.gz.")
        self.img_data = opened_img
        return opened_img

    def SkeletonTHFinder(self, nifty_file):
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
        Imin_range = np.arange(150,250, 14)
        best_result = np.inf
        img_res = None
        for i_min in Imin_range:
            self.SegmentationByTH(nifty_file, i_min, 1300)
            labels, cmp = label(self.img_data, return_num=True)
            print(f"cmp is {cmp}")
            if best_result > cmp:
                best_result = cmp
                img_res = self.img_data
        return img_res

    def _calculatePelvicLayersBoundaries(self):
        pass

    def _slice_process(self):
        out = np.zeros(self.img_data.shape)
        for slice in range(self.bottom_bound_slice, self.top_bound_slice):
            frame = self.img_data[:, :, slice]
            frame = cv2.equalizeHist(frame.astype(np.uint8))
            out[:, :, slice] = frame
        return out



        # Step 3: Get a general ROI
    def get_Nlargest_component(self):
        labels = label(self.bones)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        largestCC_img = self.bones*largestCC
        largestCC_img = skimage.morphology.closing(largestCC_img)
        final_image = nib.Nifti1Image(largestCC_img, self.raw_img.affine)
        nib.save(final_image, "out_largestCC.nii.gz")

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
    ob = PreProcessor(nifti_file='out_largestCC.nii.gz')
    # ob.get_pelvis_ROI()
    # ob.get_Nlargest_component()

