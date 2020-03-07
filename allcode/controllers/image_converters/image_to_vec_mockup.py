from allcode.controllers.image_converters.image_to_vec_converter import ImageToVecConverter
from allcode.models.image_representation.cat_or_dog_image import CatOrDogImage


class ImageToVecMockup(ImageToVecConverter):

    def convert_images(self, images):
        pass

    def convert_image(self, image):
        return CatOrDogImage(None)
