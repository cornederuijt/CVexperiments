import unittest
from allcode.controllers.app_controller.simple_image_manager import SimpleImageManager


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self._image_loc = "./data/cat_dog_images"
        self._k_in_knn = 5
        self._simpleImageManager = SimpleImageManager(self._image_loc, self._k_in_knn)
        self._test_image = ""

    def test_image_manager_mockup(self):
        # Just tests whether the application does not throw errors
        self._simpleImageManager.processes_image(self._test_image)


if __name__ == '__main__':
    unittest.main()
