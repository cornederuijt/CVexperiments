import pandas as pd
import numpy as np


class SIFTVBoW:
    def __init__(self, power_transformer=None, kmeans_model=None, classifier=None, l1_norm_sums=None):
        self._power_transformer = power_transformer
        self._kmeans_model = kmeans_model
        self._classifier = classifier
        self._l1_norm_sums = l1_norm_sums

    @property
    def power_transformer(self):
        return self._power_transformer

    @power_transformer.setter
    def power_transformer(self, power_transformer):
        self._power_transformer = power_transformer

    @property
    def k_means_model(self):
        return self._k_means_model

    @k_means_model.setter
    def k_means_model(self, k_means_model):
        self._k_means_model = k_means_model

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        self._classifier = classifier

    @property
    def l1_norm_sums(self):
        return self._l1_norm_sums

    @l1_norm_sums.setter
    def l1_norm_sums(self, l1_norm_sums):
        self._l1_norm_sums = l1_norm_sums

    def get_vector_representation(self, image_keypoint_matrix, keypoint_indices):
        image_mat_scaled = self._power_transformer.transform(image_keypoint_matrix)

        k_means_clusters = pd.concat([pd.Series(keypoint_indices, name="image_id"),
            pd.Series(np.argmin(self._kmeans_model.transform(image_mat_scaled), axis=1), name="cluster")], axis=1)

        # Compute the cluster histogram per image over the clusters
        cluster_freq_per_image = k_means_clusters\
            .groupby("image_id")['cluster']\
            .value_counts()\
            .rename("clust_counts")\
            .reset_index()

        # To matrix form and normalize:
        image_mat = cluster_freq_per_image\
            .pivot(index='image_id', columns='cluster', values='clust_counts')\
            .to_numpy()
        image_mat = np.nan_to_num(image_mat)

        image_bin_normalized = image_mat/self._l1_norm_sums

        return image_bin_normalized

    def classify_images(self, image_keypoint_matrix, keypoint_indices_distinct):
        images_as_vectors = self.get_vector_representation(image_keypoint_matrix, keypoint_indices_distinct)
        predictions_prob = self._classifier.predict_proba(images_as_vectors)[:, 0]
        predictions = self._classifier.predict(images_as_vectors)
        res = {}
        for i in range(len(keypoint_indices_distinct)):
            res[keypoint_indices_distinct[i]] = {'class': predictions[i],
                                                 'probability': predictions_prob[i]}
        return res

