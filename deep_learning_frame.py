import graphlab as gl
import os
import pandas as pd


def deep_learning(train_data, test_data):
    train_data_set = gl.SFrame(train_data)
    test_data_set = gl.SFrame(test_data)
    #train_data_set['image'] = gl.image_analysis.resize(train_data_set['image'], 48, 48, 1)
    #test_data_set['image'] = gl.image_analysis.resize(test_data_set['image'], 48, 48, 1)
    #net = gl.deeplearning.create(train_data_set, target='label')
#    pnet = gl.deeplearning.MultiLayerPerceptrons(num_hidden_layers=3,
#                                                num_hidden_units=[200, 100, 32])

    #net = gl.deeplearning.ConvolutionNet(num_convolution_layers=1,
    #                                  kernel_size=5,
    #                                  stride=2,
    #                                  num_channels=8,
    #                                  num_output_units=32)

    training_data_, validation_data = train_data_set.random_split(0.8)
    net = gl.deeplearning.create(training_data_, target='label')
    net.layers[0].kernel_size = 3
    net.layers[0].stride = 2
    net.layers[0].num_channels  = 20
    net.layers.insert(1, gl.deeplearning.layers.MaxPoolingLayer(kernel_size=2, stride=2))
    net.layers.insert(2, gl.deeplearning.layers.LocalResponseNormalizationLayer())
    net.layers.insert(3,gl.deeplearning.layers.ConvolutionLayer(kernel_size=2, stride=1,
                                                                num_channels=20))
    net.layers.insert(4, gl.deeplearning.layers.MaxPoolingLayer(kernel_size=2, stride=2))
    net.layers.insert(5, gl.deeplearning.layers.LocalResponseNormalizationLayer())
    net.layers.insert(6,gl.deeplearning.layers.ConvolutionLayer(kernel_size=3, stride=2,
                                                                num_channels=60))
    net.layers[7].kernel_size = 2
    net.layers[7].stride = 2
    net.layers.insert(8, gl.deeplearning.layers.LocalResponseNormalizationLayer())
    #net.layers.insert(2, gl.deeplearning.layers.ConvolutionLayer(kernel_size=6, stride=2,
    #                                                    num_channels=30))
    #net.layers.insert(3, gl.deeplearning.layers.MaxPoolingLayer(kernel_size=3, stride=2))
    net.layers[10].num_hidden_units = 1000
    net.layers.insert(12,gl.deeplearning.layers.FullConnectionLayer(200))
    net.layers.insert(13,gl.deeplearning.layers.RectifiedLinearLayer())
    net.layers.insert(14,gl.deeplearning.layers.DropoutLayer(0.5))
    #net.layers.insert(15,gl.deeplearning.layers.FullConnectionLayer(1200))
    #net.layers.insert(16,gl.deeplearning.layers.RectifiedLinearLayer())
    #net.layers.insert(17,gl.deeplearning.layers.DropoutLayer(0.5))
    #net.params['learning_rate'] = 0.001
    #net.params['learning_rate_schedule'] = 'exponential_decay'
    net.params['momentum'] = 0.9
    del net.layers[14]
    del net.layers[13]
    del net.layers[12]
    del net.layers[5]
    del net.layers[4]
    del net.layers[3]
    #print net.layers,net.params,net.verify()
    model = gl.neuralnet_classifier.create(training_data_, target='label',
            metric=['accuracy', 'recall@2'], max_iterations=4, random_crop=True, network=net)
    training_data_['features'] = model.extract_features(training_data_, 7)
    validation_data['features'] = model.extract_features(validation_data, 7)
    training_data_.save('train', format='csv')
    validation_data.save('val', format='csv')
    #print validation_data['features'][0]
    m = gl.boosted_trees_classifier.create(training_data_,features = ['features'], target='label', max_iterations=10)
    m.predict(validation_data)
    results = model.evaluate(validation_data)
    print results
    results = m.evaluate(validation_data)
    print results
    #pred = model.classify(test_data_set)
    #pred.remove_columns(['row_id', 'score'])
    #pred.save('deep_result', format='csv')
    #results = model.evaluate(validation_data)
    print None

def import_train_data(filename):
    data = {}
    document_dir,label_list = get_filename(filename)
    image_list = []
    for i in xrange(len(document_dir)):
        file_locate = filename +'/'+document_dir[i]
        image_list.append(gl.Image(file_locate))
        # value_dic.setdefault(document_dir[i].strip().split('_')[0],label_list[i])
    data.setdefault('label',label_list)
    data.setdefault('image',image_list)
    df = pd.DataFrame(data = data)
    return df
def import_test_data(filename):
    document_dir,label_list = get_filename(filename)
    document_dir.sort(key=lambda i: int(i.strip().split('_')[0]))
    data = {}
    image_list = []
    label_list = []
    for i in xrange(len(document_dir)):
        file_locate = filename +'/'+document_dir[i]
        image_list.append(gl.Image(file_locate))
        label_list.append(-1)
    data.setdefault('label',label_list)
    data.setdefault('image',image_list)
    df = pd.DataFrame(data = data)
    return df

def get_filename(rootDir):
    document_dir = []
    label_list = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        document_dir.append(lists)
        label_list.append(int(lists.strip().split('_')[1].split('.')[0]))
        if os.path.isdir(path):
            get_filename(path)
    return document_dir,label_list

def main():
    train_data = import_train_data('./data/image/img')
    test_data = import_test_data('./data/image/timg')
    deep_learning(train_data,test_data)
if __name__ == '__main__':
    main()
