from allcode.models.im_to_vec_models.SIFTVBoW import SIFTVBoW
from sklearn.cluster import KMeans
from sklearn import preprocessing
import multiprocessing as mp
import pickle as pl
import numpy as np
import pandas as pd


# TODO: Just make this part of the SIFTBoW class
class SIFTModelUpdater:

    def update_and_store_model(self, image_matrix, image_indices, classes, model_store_loc, random_state, k_in_kmeans,
                               xgb_empty_model, image_loc):
        image_power_transformer = preprocessing.PowerTransformer().fit(image_matrix)
        image_mat_scaled = image_power_transformer.transform(image_matrix)

        kmeans_model = KMeans(n_clusters=k_in_kmeans,
                              max_iter=1000,
                              random_state=random_state,
                              n_jobs=mp.cpu_count()-1)\
                        .fit(image_mat_scaled)

        # Transform computes distances to cluster centroids, take min to assign to a cluster
        kmeans_cluster_res = pd.concat([pd.Series(image_indices, name="image_id"),
            pd.Series(np.argmin(kmeans_model.transform(image_mat_scaled), axis=1), name="cluster")], axis=1)

        # Compute the cluster histogram per image over the clusters
        cluster_freq_per_image = kmeans_cluster_res\
            .groupby("image_id")['cluster']\
            .value_counts()\
            .rename("clust_counts")\
            .reset_index()

        # To matrix form and normalize:
        image_mat = cluster_freq_per_image\
            .pivot(index='image_id', columns='cluster', values='clust_counts')\
            .to_numpy()
        image_mat = np.nan_to_num(image_mat)

        image_mat_col_sums = image_mat.sum(axis=0)
        image_bin_normalized = preprocessing.Normalizer(norm="l1").fit_transform(image_mat)

        # fit the xgb model:
        xgb_model = xgb_empty_model.fit(image_bin_normalized, classes)

        siftbow_model = SIFTVBoW(image_power_transformer, kmeans_model, xgb_model, image_mat_col_sums)

        pl.dump(siftbow_model, open(model_store_loc, "wb"))

        image_bin_normalized_pd = pd.DataFrame(image_bin_normalized)
        image_bin_normalized_pd['image_loc'] = image_loc
        image_bin_normalized_pd.to_csv("./stored_models/data_vectorized.csv", index=False)

