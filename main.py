# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import glob
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from numpy import genfromtxt
from scipy import spatial


def extractFramesFromVideos(videoFolder):
    videos = glob.glob(os.path.join(videoFolder,"*.mp4"))
    count =0
    framesPath = os.path.join(videoFolder,"frames")
    for video in videos:
        frameExtractor(video,framesPath, count)
        count = count + 1

def extractFeatureVector(framesPath,filename):
    frames = glob.glob(os.path.join(framesPath,"*.png"))
    frames.sort()
    vectors = list()
    videos = list()
    model = HandShapeFeatureExtractor.get_instance()
    for fm in frames:
        image = cv2.imread(fm)
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        res = model.extract_feature(image)
        res = np.squeeze(res)

        vectors.append(res)
        videos.append(os.path.basename(fm))
    np.savetxt(filename, vectors, delimiter=",")

def findGesture(trainingDataFilename, testDataFilename):
    trainingData = genfromtxt(trainingDataFilename, delimiter=',')
    testData = genfromtxt(testDataFilename, delimiter=',')

    result = list()

    for vector in testData:
        cosDiff = []
        for trainingDataVector in trainingData:
            cosDiff.append(spatial.distance.cosine(vector,trainingDataVector))
        gesture = cosDiff.index(min(cosDiff))+1
        result.append(gesture)

    np.savetxt('Results.csv', result, delimiter=",", fmt='% d')

if __name__ == "__main__":
    trainingDataFolder = "traindata"
    extractFramesFromVideos(trainingDataFolder)
    framesPath = os.path.join(trainingDataFolder,"frames")
    trainingVectorFileName = "training_vector.csv"
    extractFeatureVector(framesPath,trainingVectorFileName)
    testDataFolder = "test"
    extractFramesFromVideos(testDataFolder)
    framesPath = os.path.join(testDataFolder,"frames")
    testVectorFileName = "test_vector.csv"
    extractFeatureVector(framesPath,testVectorFileName)

    findGesture(trainingVectorFileName,testVectorFileName)