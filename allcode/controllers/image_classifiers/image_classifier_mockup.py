from allcode.controllers.image_classifiers.image_classifier import ImageClassifier


class ImageClassifierMockup(ImageClassifier):

    def classify_images(self, images):
        pass

    def classify_image(self, image):
        return {'final_class': 'dog',
                'final_prob': .8}
