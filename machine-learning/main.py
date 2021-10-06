import os
import requests
import base64
from math import sqrt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

global embed
embed = hub.KerasLayer(os.getcwd()+"/../MobileNet")


class TensorVector(object):
    """helper class to process base64 encoded images into feature list vectors
    Args:
        base64data (str): base64 string to be decoded into an image and vectorised
    """
    def __init__(self, base64data=None):
        self.base64data = base64data

    def process(self):
        """processes base64data into a vector using MobileNet architecture
        """
        imgdata = base64.b64decode(self.base64data)
        img = tf.io.decode_jpeg(imgdata, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        feature_list = feature_set.tolist()
        return feature_list


def get_as_base64(url):
    """converts image in the given url to base64 string

    Args:
        url (str): url of the image to be encoded in base64

    Returns:
        str: base64 string of the given image url
    """
    res = base64.b64encode(requests.get(url).content)
    base64data = res.decode("UTF-8")
    return base64data


def cosineSim(a1,a2):
    """finds cosine similarity of given two vectors

    Args:
        a1 (list): first vector
        a2 (list): second vector

    Returns:
        float: cosine similarity of the two vectors
    """
    summed = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        summed += i*j
    cosine_sim = summed / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim
    

def vectorize_img(img_link):
    """vectoises a given base64 string of an image into its feature list

    Args:
        img_link (str): image link to be vectorized

    Returns:
        list: feature list of the given image
    """
    img = get_as_base64(img_link)
    helper = TensorVector(img)
    vector = helper.process()
    return vector


def vector_avg(avg_vector, img_vector, likes):
    """finds the weighted average of two vectors

    Args:
        avg_vector (list): avg vector of all the past liked pictures
        img_vector (list): vector of image to be averaged in
        likes (int): number of likes received after updating

    Returns:
        list: new wieghted average of the two vectors
    """
    np_avg_vector = np.array(avg_vector)
    np_img_vector = np.array(img_vector)
    new_np_avg_vector = ((likes)*np_avg_vector + np_img_vector)/(likes + 1)
    new_avg_vector =  new_np_avg_vector.tolist()
    return new_avg_vector


def weekly_list(avg_vector, img_link, threshold = 0.65):
    """checks whether a image is similar enough to be added to the weekly list

    Args:
        avg_vector (list): avg vector of all the past liked pictures
        img_link (str): image link to be checked for addition in weekly list
        threshold (float, optional): threshold value of cosine similarity to be cleared

    Returns:
        bool: should the given image be added to the weekly list or no
    """
    vector = vectorize_img(img_link)
    score = cosineSim(vector, avg_vector)
    if score > threshold:
        return True
    return False
