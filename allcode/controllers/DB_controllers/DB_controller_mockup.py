import pandas as pd
import numpy as np
import os
from allcode.controllers.DB_controllers import DBController


class DBControllerMockup(DBController):
    def __init__(self, image_sub_dir):
        super().__init__(image_sub_dir)

    def get_knn(self, image, k):
        # Actual image not use here, just draw K random images
        images = os.listdir(self._image_sub_dir)
        images = ["./data/cat_dog_images/"+im for im in images]
        if k > len(images):
            raise ValueError("K is larger than the number of cat and dog images, (which is" + str(len(images)) + ").")

        k_nearest_images = np.random.choice(images, k)
        knn_res = pd.DataFrame.from_dict(dict(zip(['k', 'image_loc'],
                                                  [np.arange(k),
                                                   k_nearest_images])))

        return knn_res

