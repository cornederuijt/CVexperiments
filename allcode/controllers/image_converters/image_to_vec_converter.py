from abc import ABC, abstractmethod


class ImageToVecConverter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def convert_images(self, images):
        pass

    @abstractmethod
    def convert_image(self, image):
        pass