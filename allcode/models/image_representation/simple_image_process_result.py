

class SimpleImageProcessResult:
    def __init__(self, final_class, final_class_prob, knn5_image_locs):
        self._final_class =final_class
        self._final_class_prob = final_class_prob
        self._knn5_image_loc = knn5_image_locs

    @property
    def final_class(self):
        return self._final_class

    @property
    def final_class_prob(self):
        return self._final_class_prob

    @property
    def knn5_image_locs(self):
        return self._knn5_image_locs
