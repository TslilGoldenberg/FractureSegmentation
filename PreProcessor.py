import numpy as np
import skimage
from skimage.measure import label
from skimage.morphology import opening, closing
import nibabel as nib
import cv2 as cv2

MAX_VOXEL_VALUE = 65535
MIN_VOXEL_VALUE = 0
FAILURE = -1
MAX_VOXEL_VALUE = 65535
MIN_VOXEL_VALUE = 0
DERIVATIVE_KERNEL = [1, -1]


class PreProcessor:
    def __init__(self, nifty_file: str):
        self.nifty_file = nifty_file
        self.raw_img = nib.load(nifty_file)
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
        MAX_VOXEL_VALUE = 65535
        MIN_VOXEL_VALUE = 0
        img = nib.load(nifty_file)
        img_data = img.get_fdata().astype(dtype=np.uint16)
        img_data[(img_data <= Imax) & (img_data > Imin)] = MAX_VOXEL_VALUE
        img_data[img_data < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
        opened_img = skimage.morphology.opening(img_data)
        final_image = nib.Nifti1Image(opened_img, img.affine)
        nib.save(final_image, f"out_seg_{Imin}_{Imax}.nii.gz.")
        self.img = opened_img
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
        Imin_range = np.arange(136, 178, 14)
        best_result = np.inf
        img_res = None
        for i_min in Imin_range:
            self.SegmentationByTH(nifty_file, i_min, 1300)
            labels, cmp = label(self.img, return_num=True)
            print(f"cmp is {cmp}")
            if best_result > cmp:
                best_result = cmp
                img_res = self.img_data
        return img_res



    # def SegmentationByTH(self, Imin, Imax):
    #     """
    #     This function is given as inputs a grayscale NIFTI file (.nii.gz) and two integers – the minimal and maximal thresholds. The function generates a segmentation NIFTI file of the same dimensions, with a binary segmentation – 1 for voxels between Imin and Imax, 0 otherwise. This segmentation NIFTI file should be saved under the name <nifty_file>_seg_<Imin>_<Imax>.nii.gz.
    #     The function returns 1 if successful, 0 otherwise. Preferably, raise descriptive errors when returning 0.
    #     :param nifty_file:
    #     :param Imin:
    #     :param Imax:
    #     :return:
    #     """
    #     self.img_data[(self.img_data <= Imax) & (self.img_data > Imin)] = MAX_VOXEL_VALUE
    #     self.img_data[self.img_data < MAX_VOXEL_VALUE] = MIN_VOXEL_VALUE
    #     opened_img = opening(self.img_data)
    #     # final_image = nib.Nifti1Image(opened_img, self.img.affine)
    #     # nib.save(final_image, f"out_seg_{Imin}_{Imax}.nii.gz.")
    #     self.img_data = opened_img

    # def SkeletonTHFinder(self):
    #     """
    #     This function iterates over 25 candidate Imin thresholds in the range of [150,500] (with intervals of 14).
    #     In each run, use the SegmentationByTH function you’ve implemented, and count the number of connectivity components
    #     in the resulting segmentation with the current Imin. Plot your results – number of connectivity components per Imin. Choose the Imin which is the first or second minima in the plot. Also, make sure to include that graph in your report.
    #
    #     Next, you need to perform post-processing (morphological operations – clean out single pixels, close holes, etc.)
    #     until you are left with a single connectivity component.
    #     Finally, this function should save a segmentation NIFTI file called “<nifty_file>_SkeletonSegmentation.nii.gz” and
    #     return the Imin used for that.
    #     :return:
    #     """
    #     Imin_range = np.arange(90, 164, 14)
    #     self.SegmentationByTH(220, 1300)
    #     best_result = np.inf
    #     img_res = None
    #     for i_min in Imin_range:
    #         img = self.img_data.copy()
    #         self.SegmentationByTH(i_min, 1300)
    #         labels, cmp = label(img, return_num=True)
    #         print(f"cmp is {cmp}")
    #         if best_result > cmp:
    #             best_result = cmp
    #             img_res = img
    #             print(i_min)
    #     return img_res

    def _calculatePelvicLayersBoundaries(self):
        pass

    def _slice_process(self):
        out = np.zeros(self.img_data.shape)
        for slice in range(self.bottom_bound_slice, self.top_bound_slice):
            frame = self.img_data[:, :, slice]
            frame = cv2.equalizeHist(frame.astype(np.uint8))
            # frame = cv2.GaussianBlur(frame, (5, 5), 1.5)
            # frame = cv2.Canny(frame, 100, 190, 3)
            # frame = cv2.dilate(frame, cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2)), 2)
            out[:, :, slice] = frame
        return out

    def _extractPelvic(self):
        # Step 1: Apply Adaptive Histogram Equalizer
        self.img_data = self._slice_process()

        # Step 2: Apply Thresholding for Skeleton
        self.bones = self.SkeletonTHFinder(self.nifty_file)
        final_image = nib.Nifti1Image(self.bones, self.raw_img.affine)
        nib.save(final_image, f"out_seg.nii.gz")


        # Step 3: Get a general ROI
    def get_Nlargest_component(self):
        labels = label(self.bones)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        final_image = nib.Nifti1Image(self.img_data*largestCC, self.raw_img.affine)
        nib.save(final_image, f"out_largestCC.nii.gz")

if __name__ == "__main__":
    ob = PreProcessor(nifty_file='datasets/case2.nii.gz')
    ob._extractPelvic()
    ob.get_Nlargest_component()
