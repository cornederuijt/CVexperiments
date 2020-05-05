from allcode.models.im_to_vec_models.SIFTVBoW import SIFTVBoW
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import multiprocessing as mp
import pickle as pl
import numpy as np
import pandas as pd


class SIFTModelUpdater:
    @staticmethod
    def run_BoW(image_matrix, image_indices, k_in_kmeans=500, kmeans_model=None,
                image_power_transformer=None, image_mat_col_sums=None):
        # TODO: Allow just for transformation as well: i.e., where all models are already provided, including the col_sums
        # Make points more "normal", required for k-means to run properly
        if image_power_transformer is None:
            image_power_transformer = preprocessing.PowerTransformer().fit(image_matrix)
        image_mat_scaled = image_power_transformer.transform(image_matrix)

        if kmeans_model is None:
            kmeans_model = KMeans(n_clusters=k_in_kmeans,
                                  max_iter=1000,
                                  n_jobs=mp.cpu_count()-1)\
                            .fit(image_mat_scaled)

        # Transform computes distances to cluster centroids, take min to assign to a cluster
        kmeans_cluster_res = pd.concat([pd.Series(image_indices, name="image_id"),
                                        pd.Series(np.argmin(kmeans_model.transform(image_mat_scaled), axis=1),
                                                  name="cluster")], axis=1)

        # Compute the cluster histogram per image over the clusters
        cluster_freq_per_image = kmeans_cluster_res \
            .groupby("image_id")['cluster'] \
            .value_counts() \
            .rename("clust_counts") \
            .reset_index()

        # To matrix form and normalize:
        image_mat = cluster_freq_per_image \
            .pivot(index='image_id', columns='cluster', values='clust_counts') \
            .to_numpy()
        image_mat = np.nan_to_num(image_mat)

        if image_mat_col_sums is None:
            image_mat_col_sums = image_mat.sum(axis=0)

        image_bin_normalized = \
            image_mat * np.repeat((1 / image_mat_col_sums), image_mat.shape[0]).\
                reshape(image_mat.shape[1], image_mat.shape[0]).transpose()

        return kmeans_model, image_power_transformer, image_bin_normalized, image_mat_col_sums

    @staticmethod
    def update_and_store_model_kmeans(image_matrix, image_indices, classes, model_store_loc, random_state,
                                      k_in_kmeans, xgb_empty_model, image_loc):
        kmeans_model, image_power_transformer, image_bin_normalized, image_mat_col_sums = \
            SIFTModelUpdater.run_BoW(image_matrix, image_indices, random_state, k_in_kmeans)

        # fit the xgb model:
        xgb_model = xgb_empty_model.fit(image_bin_normalized, classes)

        siftbow_model = SIFTVBoW(image_power_transformer, kmeans_model, xgb_model, image_mat_col_sums)

        pl.dump(siftbow_model, open(model_store_loc, "wb"))

        image_bin_normalized_pd = pd.DataFrame(image_bin_normalized)
        image_bin_normalized_pd['image_loc'] = image_loc
        image_bin_normalized_pd.to_csv("./stored_models/data_vectorized.csv", index=False)




