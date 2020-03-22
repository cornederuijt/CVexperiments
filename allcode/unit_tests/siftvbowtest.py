import unittest
from allcode.controllers.image_converters.SIFT_image_to_vec import SIFTImageToVecConverter
from allcode.controllers.DB_controllers.DB_controller_csv import DBControllerCSV
import pickle as pl
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_vectorize_image(self):
        siftbowmodel = pl.load(open("./stored_models/SIFTmodel.pl", "rb"))
        test_image_loc = "./data/cat_dog_images/cat.10.jpg"
        image_converter = SIFTImageToVecConverter()
        image_keypoint_mat = image_converter.get_keypoint_matrix_from_image_loc(test_image_loc)
        res1 = siftbowmodel.get_vector_representation(image_keypoint_mat, np.zeros(image_keypoint_mat.shape[0]))
        res2 = siftbowmodel.classify_images(image_keypoint_mat, [0])

    def test_db_controller_CV(self):
        test_image_loc = "./data/cat_dog_images/cat.10.jpg"
        K = 5
        image_converter = SIFTImageToVecConverter()
        csv_db_controller = DBControllerCSV("./data/cat_dog_images")
        siftbowmodel = pl.load(open("./stored_models/SIFTmodel.pl", "rb"))

        image_keypoint_mat = image_converter.get_keypoint_matrix_from_image_loc(test_image_loc)
        image_vec_rep = siftbowmodel.get_vector_representation(image_keypoint_mat, np.zeros(image_keypoint_mat.shape[0]))

        knn_res = csv_db_controller.get_knn(image_vec_rep, K)



if __name__ == '__main__':
    unittest.main()
