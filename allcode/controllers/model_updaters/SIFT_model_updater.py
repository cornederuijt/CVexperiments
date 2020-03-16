from allcode.models.im_to_vec_models.SIFTVBoW import SIFTVBoW
from sklearn.cluster import KMeans
from sklearn import preprocessing
import multiprocessing as mp
import pickle as pl


class SIFTModelUpdater:

    def update_and_store_model(self, image_matrix, image_indices, classes, model_store_loc, random_state, k_in_kmeans,
                               xgb_empty_model):
        image_power_transformer = preprocessing.PowerTransformer().fit(image_matrix)
        image_mat_scaled = image_power_transformer.transform(image_matrix)

        kmeans_model = KMeans(n_clusters=k_in_kmeans,
                              max_iter=1000,
                              random_state=random_state,
                              n_jobs=mp.cpu_count()-1)\
                        .fit(image_mat_scaled)
        # Check how to convert to one hot encoding
        kmeans_cluster_mat = kmeans_model.transform(image_matrix)

        # fit the xgb model:
        xgb_model = xgb_empty_model.fit(kmeans_cluster_mat, classes)

        siftbow_model = SIFTVBoW(image_power_transformer, kmeans_model, xgb_model)

        pl.dump(siftbow_model, open(model_store_loc, "rb"))

