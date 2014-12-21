#!/usr/bin/env python
# encoding: utf-8

from PIL import Image, ImageOps
import glob, os
import pandas as pd

def create_nor_data(image_path,x_size,y_size):
    data = []
    for infile in glob.glob(image_path):
        print infile
        file, ext = os.path.splitext(infile)
        y = file.split('_')[1]
        im = Image.open(infile)
        thumb = ImageOps.fit(im, (x_size,y_size), Image.ANTIALIAS)
        outfile = 'image/nor/'+ file.split('/')[1] + '/' + file.split('/')[2]
        thumb.save(outfile,'JPEG')
        image_data = [l[0]/255.0 for l in list(thumb.getdata())]
        image_data.append(im.size[0])
        image_data.append(im.size[1])
        image_data.append(y)
        data.append(image_data)
    return pd.DataFrame(data)

#data_output = pd.DataFrame(data)
#data_output.to_csv('data/train_nor_30.txt', index=False, header=False)



