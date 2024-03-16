"""Module for making datasets for UMAP/tsne"""
import csv
import os
import pickle
import time
import cv2

ALL_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']
TEST_NO = {
    'ball': [1, 7, 9],
    'spoon': [5, 7, 8],
    'mug': [12, 13, 14],
    'cup': [12, 13, 15],
    'giraffe': [1, 5, 13],
    'horse': [1, 10, 15],
    'cat': [4, 9, 15],
    'duck': [5, 9, 13],
    'helicopter': [5, 10, 15],
    'airplane': [2, 6, 15],
    'truck': [2, 6, 8],
    'car': [6, 11, 13],
}

VAL_NO = {
    'airplane': [30, 29, 28],
    'ball': [30, 29, 28],
    'car': [30, 29, 28],
    'cat': [30, 29, 28],
    'cup': [30, 29, 28],
    'duck': [30, 29, 28],
    'giraffe': [30, 29, 28],
    'helicopter': [30, 29, 28],
    'horse': [30, 29, 28],
    'mug': [30, 29, 28],
    'spoon': [30, 29, 28],
    'truck': [30, 29, 28]
}
size = 224
show = False


def crop_image(im_path, image, l, t, w, h):
    """Crop image to shape"""
    if h < w:
        t_new = max(t - ((w - h) // 2), 0)
        h_new = min(image.shape[0], w)
        l_new = l
        w_new = w
        b_new = t_new + h_new
        if b_new > image.shape[0]:
            t_new = t_new - (b_new - image.shape[0])
    elif w < h:
        t_new = t
        h_new = h
        l_new = max(l - ((h - w) // 2), 0)
        w_new = min(image.shape[1], h)
        r_new = l_new + w_new
        if r_new > image.shape[1]:
            l_new = l_new - (r_new - image.shape[1])
    else:
        t_new = t
        h_new = h
        l_new = l
        w_new = h
    
    try:
        image_cropped = image[t_new:t_new + h_new, l_new:l_new + w_new]
    except ValueError:
        print(l, t, w, h)
        return None
    try:
        assert ((image_cropped.shape[1] == image_cropped.shape[0]) or w > image.shape[0])
    except AssertionError:
        print(im_path, l, w, t, h, l_new, w_new, t_new, h_new, image_cropped.shape[1], image_cropped.shape[0])
    if show:
        cv2.imshow("image", image)
        cv2.imshow("image cropped", image_cropped)
        cv2.waitKey(0)
    return image_cropped


def generate_data_toybox(classes, num_objects, cropFilePaths, imageDir, im_out_path):
    """Generate the dataset for toybox"""
    curr = time.time()
    count = 0
    count_miss = 0
    for fp in cropFilePaths:
        cropCSVFile = list(csv.DictReader(open(fp, "r")))
        for i in range(len(cropCSVFile)):
            row = cropCSVFile[i]
            cl = row['ca']
            obj = row['no']
            tr = row['tr']
            fr = row['fr']
            try:
                left, top, width, height = int(float(row['left'])), int(float(row['top'])), int(float(row['width'])), \
                                           int(float(row['height']))
            except ValueError:
                count_miss += 1
                continue
            if cl in classes and int(obj) <= num_objects:
                os.makedirs(im_out_path + cl + "_" + obj.zfill(2) + "//", exist_ok=True)
                fileName = cl + "_" + obj.zfill(2) + "//" + cl + "_" + obj.zfill(2) + "_pivothead_" + tr + \
                                ".mp4_" + fr.zfill(3) + ".jpeg"
                imFilePath = imageDir + fileName
                imSavePath = im_out_path + fileName
                try:
                    assert os.path.isfile(imFilePath)
                except AssertionError:
                    count_miss += 1
                    print(imFilePath)
                else:
                    im = cv2.imread(filename=imFilePath)
                    im_cropped = crop_image(im_path=imFilePath, image=im, l=left, t=top, h=height, w=width)
                    im_resized = cv2.resize(im_cropped, (size, size), interpolation=cv2.INTER_CUBIC)
                    assert cv2.imwrite(imSavePath, im_resized)
                    count += 1
            if i % 1000 == 0:
                print(i, time.time() - curr)
        print("Time taken for file", fp, ":", time.time() - curr)
        curr = time.time()
    print("Missed files:", count_miss)


crop = True
numObjectsInClass = 30
toyboxFramesPath = "/media/sanyald/AIVAS/Toybox_frames/Toybox_New_Frame6_Size1920x1080/"
out_path = "../data/umap_data_3fps/"
os.makedirs(out_path, exist_ok=True)
imOutPath = out_path + "images/"

csvFiles = ["../data/toybox_rot_frames_interpolated.csv"]
generate_data_toybox(classes=ALL_CLASSES, num_objects=numObjectsInClass, imageDir=toyboxFramesPath,
                     cropFilePaths=csvFiles, im_out_path=imOutPath)

