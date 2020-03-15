
class SIFTVBoW:
    def __init__(self, power_transformer=None, kmeans_model=None, classifier=None):
        self._power_transformer = power_transformer
        self._kmeans_model = kmeans_model
        self._classifier = classifier

    @property
    def power_transformer(self):
        return self._power_transformer

    @power_transformer.setter
    def power_transformer(self, power_transformer):
        self._power_transformer = power_transformer

    @property
    def k_means_model(self):
        return self._k_means_model

    @k_means_model.setter
    def k_means_model(self, k_means_model):
        self._k_means_model = k_means_model

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        self._classifier = classifier
