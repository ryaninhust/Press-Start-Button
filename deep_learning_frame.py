import graphlab as gl
import os
import pandas as pd

def deep_learning(train_data,test_data):
	print '!!!'
	print train_data,test_data
	train_data_set = gl.SFrame(train_data)
	test_data_set = gl.SFrame(test_data)
	'''net = gl.deeplearning.create(train_data_set, target='label')'''
	net = gl.deeplearning.ConvolutionNet(num_convolution_layers=2,
                                               kernel_size=3, stride=2,
                                               num_channels=3,
                                               num_output_units=32)
	training_data_, validation_data = train_data_set.random_split(0.8)
	print net.layers,net.params,net.verify()	
	model = gl.neuralnet_classifier.create(training_data_, target='label',network=net,validation_set=validation_data,
			metric=['accuracy', 'recall@2'])
	pred = model.classify(test_data_set)
	results = model.evaluate(test_data_set)
	print results

def import_train_data(filename):
	value_dic = {}
	data = {}
	document_dir,label_list = get_filename(filename)
	image_list = []
	for i in xrange(len(document_dir)):
		file_locate = filename +'/'+document_dir[i]
		image_list.append(gl.Image(file_locate))
		#value_dic.setdefault(document_dir[i].strip().split('_')[0],label_list[i])
	data.setdefault('label',label_list)
	data.setdefault('image',image_list)
	df = pd.DataFrame(data = data)
	return df
def import_test_data(filename):
	document_dir,label_list = get_filename(filename)
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
	train_data = import_train_data('./data/image/nor/train')
	test_data = import_test_data('./data/image/nor/test')
	deep_learning(train_data,test_data)
if __name__ == '__main__':
	main()