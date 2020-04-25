from allcode.controllers.image_converters.image_to_vec_converter import ImageToVecConverter
import cv2
from deprecated import deprecated
import pandas as pd
import multiprocessing as mp
import numpy as np


class SIFTImageToVecConverter(ImageToVecConverter):
    def __init__(self, siftvbow_model=None):
        super().__init__()
        self._siftvbow_model = siftvbow_model
        self._sift_model = cv2.xfeatures2d.SIFT_create()

    @deprecated("Not implemented")
    def convert_images(self, images):
        image_indices = []
        for i in range(len(images)):
            image_indices.append(np.repeat(i, images[i].shape[0]))

        image_keypoint_matrix = np.vstack(images)
        image_indices = np.hstack(image_indices)

        image_bin_normalized = self._keypoint_to_image_vec(image_keypoint_matrix, image_indices)

        return image_bin_normalized

    def convert_image(self, image):
        image_keypoint_matrix = self._get_keypoint_matrix(image)
        keypoint_indices = np.zeros(image_keypoint_matrix.shape[0])

        image_bin_normalized = self._keypoint_to_image_vec(image_keypoint_matrix, keypoint_indices)

        return image_bin_normalized

    def get_keypoint_matrix_multi_image_location(self, image_locations):
        # TODO: parallize
        image_matrices = [self._get_keypoint_matrix_from_image_loc(im_loc) for im_loc in image_locations]

        image_indices = []
        for i in range(len(image_matrices)):
            image_indices.append(np.repeat(i, image_matrices[i].shape[0]))

        image_large_matrix = np.vstack(image_matrices)
        image_indices = np.hstack(image_indices)

        return image_indices, image_large_matrix

    def _keypoint_to_image_vec(self, image_keypoint_matrix, keypoint_indices):
        image_mat_scaled = self._siftvbow_model.power_transformer.transform(image_keypoint_matrix)

        k_means_clusters = pd.concat([pd.Series(keypoint_indices, name="image_id"),
                                      pd.Series(np.argmin(self._siftvbow_model
                                                              .kmeans_model
                                                              .transform(image_mat_scaled), axis=1), name="cluster")],
                                     axis=1)

        # Compute the cluster histogram per image over the clusters
        cluster_freq_per_image = k_means_clusters \
            .groupby("image_id")['cluster'] \
            .value_counts() \
            .rename("clust_counts") \
            .reset_index()
        # To matrix form and normalize:
        image_mat = cluster_freq_per_image \
            .pivot(index='image_id', columns='cluster', values='clust_counts') \
            .to_numpy()

        image_mat = np.nan_to_num(image_mat)
        image_bin_normalized = image_mat / self._siftvbow_model.l1_norm_sums

        return image_bin_normalized

    def _get_keypoint_matrix_from_image_loc(self, image_location):
        return self._get_keypoint_matrix(cv2.imread(image_location))

    def _get_keypoint_matrix(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, keypoint_hist = self._sift_model.detectAndCompute(gray, None)

        return keypoint_hist

