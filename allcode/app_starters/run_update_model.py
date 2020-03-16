from allcode.controllers.model_updaters.SIFT_model_updater import SIFTModelUpdater
from allcode.controllers.image_converters.SIFT_image_to_vec import SIFTImageToVecConverter
from xgboost.sklearn import XGBClassifier
import multiprocessing as mp
from os import listdir
import pandas as pd
import numpy as np


if __name__ == '__main__':

    # 1) Load images and create image matrix
    # TODO: For testing purposes just taking 30 images of each category
    # TODO: Change this into a unit test

    model_store_loc = "./stored_models/SIFTmodel.pl"
    rand_state = np.random.RandomState(1992)
    k_in_kmeans = 5

    images_loc = listdir("./data/cat_dog_images")
    images_loc = ["./data/cat_dog_images/"+s for s in images_loc]
    images_pd = pd.DataFrame.from_dict(dict(zip(['images'], [images_loc])))
    cat_images = images_pd.loc[images_pd['images'].str.contains('cat', na=False), 'images'].head(30).tolist()
    dog_images = images_pd.loc[images_pd['images'].str.contains('dog', na=False), 'images'].head(30).tolist()

    test_images = cat_images + dog_images

    # Update the SIFT model
    sift_model_updater = SIFTModelUpdater()
    im_to_vec_converter = SIFTImageToVecConverter()

    keypoint_indices, keypoint_mat = im_to_vec_converter.get_keypoint_matrix_multi_image_location(test_images)

    # Dog = 1, cat = 0
    classes = np.hstack((np.zeros(30), np.ones(30)))

    # TODO: Following should be loaded from a file (these are default settings)
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=mp.cpu_count()-1,
        scale_pos_weight=1,
        seed=27)

    # self, image_matrix, classes, model_store_loc, random_state, k_in_kmeans, xgb_empty_model
    sift_model = sift_model_updater.update_and_store_model(keypoint_mat, keypoint_indices, classes, model_store_loc,
                                                           rand_state, k_in_kmeans, xgb1)







