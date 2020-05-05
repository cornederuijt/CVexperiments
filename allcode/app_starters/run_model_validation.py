from os import listdir
import pandas as pd
import numpy as np
from allcode.model_development.SIFT_model_validation import SIFTModelValidator


if __name__ == '__main__':
    model_store_loc = "./stored_models/SIFTmodel.pl"
    rand_state = np.random.RandomState(1992)
    k_in_kmeans = 5

    images_loc = listdir("./data/cat_dog_images")
    images_loc = ["./data/cat_dog_images/"+s for s in images_loc]
    images_pd = pd.DataFrame.from_dict(dict(zip(['images'], [images_loc])))
    cat_images = images_pd.loc[images_pd['images'].str.contains('cat', na=False), 'images'].head(30).tolist()
    dog_images = images_pd.loc[images_pd['images'].str.contains('dog', na=False), 'images'].head(30).tolist()

    test_images = cat_images + dog_images

    classes = np.hstack((np.zeros(30), np.ones(30)))

    SIFTModelValidator.validate_models(test_images, classes)


