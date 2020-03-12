from allcode.controllers.image_converters.SIFT_image_to_vec import SIFTImageToVecConverter
import multiprocessing as mp
import numpy as np
from sklearn.cluster import KMeans


class SIFTKMeansUpdater:
    def __init__(self):
        # Of course do not provide a model, as we are defining one here
        self._sift_converter = SIFTImageToVecConverter()

    def update_model(self, train_images, model_store_loc, random_state):
        pool = mp.Pool(mp.cpu_count() - 1)
        image_matrix_list = [pool.apply(self._sift_converter.get_keypoint_matrix,
                             args=image) for image in train_images]
        pool.close()

        # TODO: Check how large this matrix becomes, 500 x 128 per image. So 100 images already gives a pretty large matrix
        # TODO: There's also mini batch k-means on SKLEARN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
        # TODO (in case memory becomes an issue), DBSCAN and OPTICS are also options which are quicker (and do not work with Euclidean distance)
        image_full_mat = np.stack(image_matrix_list, axis=0)
        self._find_optimal_clusters_AIC(image_full_mat, random_state)

    def _find_optimal_clusters_AIC(self, image_mat, k_grid, random_state):
        sse_list = []
        k_means_model_best = None
        sse_best = np.inf
        for k in k_grid:
            k_means_model = KMeans(n_clusters = k, max_iter=1000, random_state=random_state.get_state()).fit(image_mat)
            sse_list = sse_list.append(k_means_model.intertia_)  # apparently this is the SSE
            if k_means_model.intertia_ < sse_best:
                sse_best = k_means_model.intertia_
                k_means_model_best = k_means_model

        return k_means_model_best, k_grid, sse_list  # Note that the SSE is included in the best model by intertia prop
