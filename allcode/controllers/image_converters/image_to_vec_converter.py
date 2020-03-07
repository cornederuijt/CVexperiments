from abc import ABC, abstractmethod


class ImageToVecConverter(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def convert_images(self, images):
        pass

    @abstractmethod
    def convert_image(self, image):
        pass