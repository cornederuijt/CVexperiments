from allcode.controllers.image_classifiers.image_classifier_mockup import ImageClassifierMockup
from allcode.controllers.image_converters import ImageToVecMockup
from allcode.models.image_representation.simple_image_process_result import SimpleImageProcessResult


class SimpleImageManager:
    def __init__(self):
        self._image_classifier = ImageClassifierMockup()
        self._image_converter = ImageToVecMockup()

    def processes_image(self, image):
        image_vec_rep = self._image_converter.convert_image(image)
        image_classifier_res = self._image_classifier.classify_image(image_vec_rep)

        res = SimpleImageProcessResult()
