from allcode.controllers.image_converters.image_to_vec_converter import ImageToVecConverter
import cv2
from deprecated import deprecated
import multiprocessing as mp
import numpy as np


class SIFTImageToVecConverter(ImageToVecConverter):
    def __init__(self, k_means_model=None):
        super().__init__()
        self._k_means_model = k_means_model
        self._sift_model = cv2.xfeatures2d.SIFT_create()

    @property
    def k_means_model(self):
        return self._k_means_model

    @k_means_model.setter
    def k_means_model(self, k_means_model):
        self._k_means_model = k_means_model

    @deprecated("Not implemented")
    def convert_images(self, images):
        pass

    @deprecated("Not implemented")
    def convert_image(self, image):
        # TODO: check format of the image, perhaps it cannot be read like this by cv2
        # Convert to grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corner_points = self._sift_model.detect(gray, None)

        # TODO: How to convert to proper matrix?

    def get_keypoint_matrix_multi_image_location(self, image_locations):
        # TODO: parallize
        image_matrices = [self.get_keypoint_matrix_from_image_loc(im_loc) for im_loc in image_locations]

        image_indices = []
        for i in range(len(image_matrices)):
            image_indices.append(np.repeat(i, image_matrices[i].shape[0]))

        image_large_matrix = np.vstack(image_matrices)
        image_indices = np.hstack(image_indices)

        return image_indices, image_large_matrix

    def get_keypoint_matrix_from_image_loc(self, image_location):
        return self.get_keypoint_matrix(cv2.imread(image_location))

    def get_keypoint_matrix(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, keypoint_hist = self._sift_model.detectAndCompute(gray, None)

        return keypoint_hist

