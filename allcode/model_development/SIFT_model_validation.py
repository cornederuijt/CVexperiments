from allcode.models.im_to_vec_models.SIFTVBoW import SIFTVBoW
import numpy as np
import numpy.random as rd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import model_selection
from allcode.controllers.image_converters.SIFT_image_to_vec import SIFTImageToVecConverter
from allcode.controllers.model_updaters.SIFT_model_updater import SIFTModelUpdater
from allcode.misc.validate_classification_models import ClassificationModelValidator


class SIFTModelValidator:

    @staticmethod
    def validate_models(images, classes, valid_frac=0.1, test_frac=0.2, default_k_in_kmeans=500, seed=1992):
        # 0) Set seed:
        np.random.seed(seed)

        # 1) Convert images to large vector
        im_to_vec_converter = SIFTImageToVecConverter()
        image_indices, keypoint_mat = im_to_vec_converter.get_keypoint_matrix_multi_image_location(images)

        # Run K_means with a default number of clusters. Following
        # https://www.researchgate.net/post/Image_classification_using_SIFT_features_and_SVM2:
        # "the 'K' parameter (the number of clusters) depends on the number of SIFTs that you have for training,
        # but usually is around 500->8000 (the higher, the better)."
        # As default value we use the minimum: 500 (btw, obviously 'the higher the better', is not completely true)

        # 2) Split images into train, validation, and test (.7, 0.1(=1-.9), and 0.2 (=1-0.7-0.1))
        train_frac = 1 - (valid_frac + test_frac)
        unique_image_indices = np.unique(image_indices)
        train_ind, val_ind, test_ind = np.split(rd.choice(unique_image_indices,
                                                   size=unique_image_indices.shape[0],
                                                   replace=False),
                                        [int(train_frac * unique_image_indices.shape[0]),
                                          int((1 - valid_frac) * unique_image_indices.shape[0])])

        # 3) Run initial K-means model
        # First train the k_means model on the training data:
        im_ind_in_train = np.isin(image_indices, train_ind)
        im_id_in_valid = np.isin(image_indices, val_ind)

        kmeans_model, image_power_transformer, image_bin_normalized_train, image_mat_col_sums = \
            SIFTModelUpdater.run_BoW(keypoint_mat[im_ind_in_train],
                                     image_indices[im_ind_in_train],
                                     k_in_kmeans=default_k_in_kmeans)

        # Run k-means on validation:
        _, _, image_bin_normalized_valid, _ = \
            SIFTModelUpdater.run_BoW(keypoint_mat[im_id_in_valid],
                                     image_indices[im_id_in_valid],
                                     kmeans_model=kmeans_model,
                                     image_power_transformer=image_power_transformer,
                                     image_mat_col_sums=image_mat_col_sums)

        # 4) Run the auto-ML: test on logistic regression, Random forest, XGBoost, SVM, and KNN
        # TODO: Check whether order of X_mat and labels are correct!
        ClassificationModelValidator.run_auto_ml(image_bin_normalized_train, image_bin_normalized_valid,
                                                 classes[train_ind], classes[val_ind])










