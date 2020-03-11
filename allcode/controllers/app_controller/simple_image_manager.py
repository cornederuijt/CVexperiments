from allcode.controllers.image_classifiers.image_classifier_mockup import ImageClassifierMockup
from allcode.controllers.image_converters.image_to_vec_mockup import ImageToVecMockup
from allcode.models.image_representation.simple_image_process_result import SimpleImageProcessResult
from allcode.controllers.DB_controllers.DB_controller_mockup import DBControllerMockup


class SimpleImageManager:
    def __init__(self, image_locations, k_in_knn):
        self._image_classifier = ImageClassifierMockup()
        self._image_converter = ImageToVecMockup()
        self._db_controller = DBControllerMockup(image_locations)
        self._k_in_knn = k_in_knn

    def processes_image(self, image):
        image_vec_rep = self._image_converter.convert_image(image)
        image_classifier_res = self._image_classifier.classify_image(image_vec_rep)
        image_knn = self._db_controller.get_knn(image, self._k_in_knn)

        res = SimpleImageProcessResult(image,
                                       image_classifier_res['final_class'],
                                       image_classifier_res['final_prob'],
                                       image_knn)

        return res
