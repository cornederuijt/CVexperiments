from allcode.models.im_to_vec_models.SIFTVBoW import SIFTVBoW
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import model_selection


class SIFTModelValidator:

    def update_model(self, image_matrix, classes, model_store_loc, random_state, train_frac=0.7, valid_frac=0.1, test_frac=0.2):
        # TODO: Check how large this matrix becomes, 500 x 128 per image. So 100 images already gives a pretty large matrix
        # TODO: There's also mini batch k-means on SKLEARN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
        # TODO (in case memory becomes an issue), DBSCAN and OPTICS are also options which are quicker (and do not work with Euclidean distance)

        X_train, y_train, X_test, y_test = model_selection\
            .train_test_split(image_matrix, classes,
                              test_size=valid_frac+test_frac,
                              random_state=random_state.get_state())

        X_test, y_test, X_valid, y_valid = model_selection\
            .train_test_split(image_matrix, classes,
                              test_size=valid_frac/(valid_frac+test_frac),
                              random_state=random_state.get_state())

        # Based on 128 size vectors for each key point (first Fibonacci number after ceil(2*sqrt(128))=23)
        k_grid = [2, 3, 5, 8, 13, 21, 34]
        k_means_model_best, power_transformer, X_train_scaled, sse_aic_list = \
            self._find_optimal_clusters_AIC(X_train, k_grid, random_state)

        siftbow_model = SIFTVBoW(power_transformer, k_means_model_best)

        return siftbow_model, image_mat_scaled, k_grid, sse_aic_list

    def _find_optimal_clusters_AIC(self, image_mat, k_grid, random_state):
        # First perform power transform and scale:
        image_power_transformer = preprocessing.PowerTransformer().fit(image_mat)
        image_mat_scaled = image_power_transformer.transform(image_mat)

        # Fit k_means for different k and compute AIC
        sse_aic_list = []
        k_means_model_best = None
        sse_best_aic = np.inf
        for k in k_grid:
            k_means_model = KMeans(n_clusters=k, max_iter=1000, random_state=random_state.get_state())\
                .fit(image_mat_scaled)
            cur_sse_aic = 2 * (k_means_model.intertia_ - k)
            sse_aic_list = sse_aic_list.append(cur_sse_aic)  # apparently this is the SSE

            if cur_sse_aic < sse_best_aic:
                sse_best_aic = cur_sse_aic
                k_means_model_best = k_means_model

        return k_means_model_best, \
               image_power_transformer, \
               image_mat_scaled, \
               sse_aic_list  # Note that the SSE is included in the best model by intertia prop
