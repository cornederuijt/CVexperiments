import pandas as pd
from allcode.controllers.DB_controllers.DBController import DBController
from sklearn.metrics import pairwise_distances
import multiprocessing as mp


class DBControllerCSV(DBController):
    def __init__(self, image_sub_dir):
        super().__init__(image_sub_dir)

    def get_knn(self, image, k):
        image_db = pd.read_csv("./stored_models/data_vectorized.csv", index_col=False)

        image_loc = image_db['image_loc']
        image_mat = image_db.drop('image_loc', axis=1).to_numpy()

        dist = pd.concat([image_loc,
            pd.Series(pairwise_distances(image_mat, image, n_jobs=mp.cpu_count()-1)
                      .reshape(image_loc.shape[0]), name="eucl_dist")],
                         axis=1)
        dist['k'] = dist['eucl_dist'].rank(method='first')

        top_k = dist.sort_values('k', axis=0).loc[:, ['k', 'image_loc']].head(k)

        return top_k

