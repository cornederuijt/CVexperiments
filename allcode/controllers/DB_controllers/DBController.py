from abc import ABC, abstractmethod


class DBController(ABC):
    def __init__(self, image_sub_dir):
        super().__init__()
        self._image_sub_dir = image_sub_dir

    @property
    def image_sub_directory(self):
        return self._image_sub_dir

    @abstractmethod
    def get_knn(self, image, k):
        pass

