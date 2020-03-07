

class CatOrDogImage:
    def __init__(self, image_representation):
        self._image_representation = image_representation

    @property
    def image_representation(self):
        return self._image_representation

    @image_representation.setter
    def image_representation(self, image_representation):
        self._image_representation = image_representation
