import marshmallow_dataclass
from dataclasses import dataclass


@dataclass
class AppConfig:
    images_loc: str
    version: str
    k_in_knn: int
