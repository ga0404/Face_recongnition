# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 21:26:39 2018

@author: louis
"""

from glob import glob
from PIL import Image
import os
#和這程式同位置創兩個資料夾，一個放原始檔，一個放修改後的
i=0
source_dir = 'source_images'
target_dir = 'resized_images'

filenames = glob('{}/*'.format(source_dir))

print(filenames)

for filename in filenames:
    with Image.open(filename) as im:
        width, height = im.size
        print(filename, width, height, os.path.getsize(filename))


#我們先用os.path.exists()函數判定這個目錄是否存在。
#當判定為不存在時，我們採用os.makedirs()函數來創建它       
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
#對需要壓縮的圖片，新的寬度和高度
for filename in filenames:
   
    filesize = os.path.getsize(filename)
    
    with Image.open(filename) as im:
            i+=1
            width, height = im.size
            new_height = 180
            
            new_width = 240
            print('adjusted size:', new_height, new_width)
            resized_im = im.resize((new_height, new_width),Image.ANTIALIAS)
            output_filename = filename.replace(source_dir, target_dir)
            if(i<10):
             resized_im.save("resized_images/410421236_0"+str(i)+".jpg")
            else:
             resized_im.save("resized_images/410421236_"+str(i)+".jpg")