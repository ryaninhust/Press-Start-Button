#!/usr/bin/env python
# encoding: utf-8

from PIL import Image, ImageOps
import glob, os
import numpy as np

def create_nor_data(image_path, x_size, y_size, normalize = False):
    data = []

    for infile in glob.glob(image_path):
        print infile
        y = os.path.basename(infile).split('.')[0].split('_')[1]
        im = Image.open(infile)
        thumb = ImageOps.fit(im, (x_size,y_size), Image.ANTIALIAS)
        #outfile = 'image/nor/'+ infile.split('/')[1] + '/' + infile.split('/')[2]
        #thumb.save(outfile,'JPEG')
        if normalize:
            image_data = [1 if l[0]/255.0 > 0 else 0 for l in list(thumb.getdata())]
        else:
            image_data = [l[0]/255.0 for l in list(thumb.getdata())]
        image_data.append(int(y))
        data.append(image_data)

    data_m = np.matrix(data)
    data_x = data_m[:, :-1]
    data_y = np.array(data_m[:, -1]).T[0]
    return data_x, data_y




