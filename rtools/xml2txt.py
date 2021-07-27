# coding=utf-8

import os
import math
import xml.dom.minidom
import cv2 as cv


def xml_to_txt(indir, outdir):
    os.chdir(indir)
    xmls = os.listdir('./')
    for i, file in enumerate(xmls):
        file_save = file.split('.')[0] + '.txt'
        file_txt = os.path.join(outdir, file_save)
        f_w = open(file_txt, 'w')
        # actual parsing
        DOMTree = xml.dom.minidom.parse(file)
        annotation = DOMTree.documentElement
        filename = annotation.getElementsByTagName("path")[0]
        imgname = filename.childNodes[0].data
        img_temp = imgname.split('\\')[-1]
        img_temp = os.path.join(image_dir, img_temp)
        # image = cv.imread(imgname)
        # cv.imwrite(img_temp, image)
        objects = annotation.getElementsByTagName("object")
        print(file)
        for object in objects:
            bbox = object.getElementsByTagName("robndbox")[0]
            cx = bbox.getElementsByTagName("cx")[0]
            x = float(cx.childNodes[0].data)
            print(x)
            cy = bbox.getElementsByTagName("cy")[0]
            y = float(cy.childNodes[0].data)
            print(y)
            cw = bbox.getElementsByTagName("w")[0]
            w = float(cw.childNodes[0].data)
            print(w)
            ch = bbox.getElementsByTagName("h")[0]
            h = float(ch.childNodes[0].data)
            print(h)
            cangel = bbox.getElementsByTagName("angle")[0]
            angle = float(cangel.childNodes[0].data)
            # angle = angle * 180 / math.pi
            print(angle)
            cname = object.getElementsByTagName("name")[0]
            name = cname.childNodes[0].data
            print(name)

            # if name == 'sh_damaged_half':
            #     cat_index = '0'
            # elif name == 'sh_damaged_comp':
            #     cat_index = '1'
            # print(cat_index)

            x1 = x - w / 2.
            y1 = y - h / 2.
            x2 = x + w / 2.
            y2 = y - h / 2.
            x3 = x + w / 2.
            y3 = y + h / 2.
            x4 = x - w / 2.
            y4 = y + h / 2.

            temp = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(x3) + ' ' + str(y3) + ' ' + str(
                x4) + ' ' + str(y4) + ' ' + name + ' ' + str(angle) + ' ' + '\n'
            f_w.write(temp)
        f_w.close()


image_dir = '/home/maggie/work/r3det-on-mmdetection/data/sh_damaged/val/images'
indir = '/home/maggie/work/r3det-on-mmdetection/data/sh_damaged/val/roxml'  # xml目录
outdir = '/home/maggie/work/r3det-on-mmdetection/data/sh_damaged/val/labelTxt'  # txt目录
xml_to_txt(indir, outdir)
