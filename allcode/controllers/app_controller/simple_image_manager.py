from allcode.controllers.image_converters.SIFT_image_to_vec import SIFTImageToVecConverter
from allcode.models.image_representation.simple_image_process_result import SimpleImageProcessResult
from allcode.controllers.DB_controllers.DB_controller_csv import DBControllerCSV
import pickle as pl
import numpy as np


class SimpleImageManager:
    def __init__(self, image_locations, k_in_knn):
        self._siftbowmodel = pl.load(open("./stored_models/SIFTmodel.pl", "rb"))
        self._image_converter = SIFTImageToVecConverter()
        self._db_controller = DBControllerCSV(image_locations)
        self._k_in_knn = k_in_knn

    def processes_image(self, image):
        image_keypoint_matrix = self._image_converter.get_keypoint_matrix(image)
        image_vec_rep = self._siftbowmodel.get_vector_representation(image_keypoint_matrix,
                                                               np.zeros(image_keypoint_matrix.shape[0]))
        image_classifier_res = self._siftbowmodel.classify_images(image_keypoint_matrix, [0])
        image_knn = self._db_controller.get_knn(image_vec_rep, self._k_in_knn)

        res = SimpleImageProcessResult(image,
                                       image_classifier_res[0]['class'],
                                       image_classifier_res[0]['probability'],
                                       image_knn)

        return res
