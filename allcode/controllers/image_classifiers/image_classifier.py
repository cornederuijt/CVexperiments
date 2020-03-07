from abc import ABC, abstractmethod


class ImageClassifier(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def classify_images(self, images):
        pass

    @abstractmethod
    def classify_image(self, image):
        pass