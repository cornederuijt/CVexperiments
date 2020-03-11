

class SimpleImageProcessResult:
    def __init__(self, image, final_class, final_class_prob, knn5_image_locations):
        self._image = image
        self._final_class = final_class
        self._final_class_prob = final_class_prob
        self._knn5_image_locations = knn5_image_locations

    @property
    def image(self):
        return self._image

    @property
    def final_class(self):
        return self._final_class

    @property
    def final_class_prob(self):
        return self._final_class_prob

    @property
    def knn5_image_locations(self):
        return self._knn5_image_locations

