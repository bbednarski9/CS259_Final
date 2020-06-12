import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import tqdm as tqdm
import time
import shutil

#sets=[('2007', 'train'), ('2007', 'val')]
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file_vocdev = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    out_file_customlabels = open('labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file_vocdev.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file_customlabels.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
#wd= '/content/gdrive/My Drive/209AS_AIML/pascal_voc/data'
time.sleep(1)
for year, image_set in tqdm.tqdm(sets, desc='First Loop'):
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in tqdm.tqdm(image_ids, desc='Second Loop'):
        #list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        list_file.write('%s/images/%s.jpg\n'%(wd, image_id))
        # copy image to custom image set
        imgpth = '%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id)
        imgdest = 'images/'
        shutil.copy(imgpth, imgdest)
        convert_annotation(year, image_id)
        time.sleep(.01)
    list_file.close()

#os.system("cat 2007_train.txt > train.txt")
os.system("cat 2007_train.txt 2012_train.txt 2012_val.txt > train.txt")
os.system("cat 2007_val.txt > valid.txt")