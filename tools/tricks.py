from utils import X_train, y_train, X_test, y_test
from PIL import Image

def convert_feauture_into_img(feature_matrix):
    for idx, feature_vec in enumerate(feature_matrix):
        feature_dense = feature_vec.todense()
        im = Image.fromarray(feature_dense.reshape((122, 105)) * 255)
        im.convert('RGB').save('data/image/test/%d.jpg' % idx)

convert_feauture_into_img(X_test)


