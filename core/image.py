from skimage.io import imread_collection
from skimage.util import img_as_float
from sklearn import cross_validation


class ImgaeCase(object):

    def __init__(self, file_name, pixel_matrix, feature_machines):
        self.file_name = file_name
        self.pixel_matrix = pixel_matrix
        self.features_machines = feature_machines
        self._features_vector = []

    @property
    def feature_vector(self):
        """Property for store feature
        """
        if not self._features_vector:
            pass
        return self._features_vector

    def extract_features(self):
        """Generate Features
        """
        for feature in self.features_machines:
            feature_vector = feature.gen_features(self)
            self._features_vector.extend(feature_vector)


class ImageCases(object):
    def __init__(self, feature_machines):
        self.image_collection = []
        self._labels = []
        self._features_matrix = []
        self.feature_machines = feature_machines

        #TODO This is Y for train/test

    def load_data(self, dir_path):
        raw_image_collection = imread_collection(dir_path + '/*.jpg')
        for index, img in enumerate(raw_image_collection):
            image_case = ImageCases(raw_image_collection.files[index],
                                    img_as_float(img),
                                    self.feature_machines,
                                    )
            self.image_collection.append(image_case)

    @property
    def feature_matrix(self):
        if not self._features_matrix:
            for img_case in self.image_collection:
                self._features_matrix.append(img_case.feature_vector)
        return self._features_matrix

    def cross_validate(self, learning_model, labels, cv):
        scores = cross_validation.cross_val_score(learning_model,
                                                  self.feature_matrix,
                                                  labels, cv)
        return scores


