import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread,imsave
import skimage.filters
from skimage import exposure
from skimage import img_as_float
import skimage.morphology
import skimage.segmentation
import skimage.transform
import PIL
import time
import shutil
import os


def processImages(path,extension):
    #Implement reading files from provided path.
    imgList = os.listdir(path=path)
    for img in imgList:
        pass

    
def plotImageHistogram(img,bins):
    axes = np.zeros((1,2),dtype=np.object)
    axes[0,0]=plt.subplot(2,1,1)
    imgAxes = axes[0,0]
    axes[0,1] = plt.subplot(2,1,2)
    histAxes = axes[0,1]
    cdfAxes = histAxes.twinx()
    image = img_as_float(image=img)
    imgAxes.imshow(img,cmap=plt.cm.gray)
    histAxes.hist(image.ravel(),bins=bins,histtype='step',color='black')
    histAxes.ticklabel_format(axis='Y',style='scientific',scilimits=(0,0))
    histAxes.set_xlabel('Pixel Intensity')

    cdf,bins = exposure.cumulative_distribution(image=img,nbins=bins)
    cdfAxes.plot(bins,cdf,'r')
    
    
    plt.show()
"""
    image: Input Image
    numChunks: Image to be brokent into these number of chunks
    hSplits: Number of splits based on image height
    vSplits: Number of splits based on image width
    chunkingType: row, column , both. Only Both implemented for now.
    overlap: Number of pixels of overlap
    persistChunks: To store chunks of images to disk

"""
def extractSubSections(image,numChunks=1,hSplits=1,vSplits=1,chunkingType='both',hOverlap=150,wOverlap=200,presistChunks=False):
    #Implement method to cut images into sections
    imgHeight,imgWidth = image.shape
    chunkHeight = int(imgHeight / hSplits)
    chunkWidth = int(imgWidth / vSplits)
    # print(imgWidth,imgHeight)
    # print(chunkWidth,chunkHeight)
    # img = image[0:chunkHeight+hOverlap,0:chunkWidth+vOverlap]
    fig, axes = plt.subplots(nrows = 3, ncols=4, figsize=(5, 5))
    ax = axes.ravel()
    
    currChunk = 0 
    img = []
    startH = startW = 0
    # img.append(image)
    for hSplit in range(0,hSplits):
        for vSplit in range(0,vSplits):
            print("Start....")
            print("HSplit: ",hSplit)
            print("VSplit: ",vSplit)
            print("StartH: ",startH)
            print("StartW: ",startW)
            endH = (hSplit + 1) * chunkHeight
            endW = (vSplit + 1) * chunkWidth
            endHOverlap = endH + hOverlap
            endWOverlap = endW + wOverlap
            print("EndH: ",endH)
            print("EndW: ",endW)
            print("End.....")
            ##Processing for first Row
            img.append(image[startH:endHOverlap,startW:endWOverlap])
            startW = endW - wOverlap
        ## Horizontally we should move to next row.
        ## Vertically reset it 0
        startH = endH - hOverlap
        startW = 0
    return img

    # subImg = image[0:1500,0:1500]
    # plt.imshow(subImg,cmap=plt.cm.gray)
    # plt.show()

def filterImage(img,filterType):
    pass

def transformImage(img, transformType):
    pass

def segmentImage(img,segmentType):
    pass



path = "C:\\DeepLearning\\Vision\\EL\\allimg"
extension = "jpg"
img = imread("C:\\DeepLearning\\Vision\\EL\\allimg\\A2FD778.jpg",as_gray=True)
# image = 255 - img
# gamm_created = exposure.equalize_adapthist(img,kernel_size=20,clip_limit=0.02,nbins=10)
# img =  exposure.adjust_gamma(gamm_created,1.5,1)
# plotImageHistogram(img,256)
# plotImageHistogram(gamm_created,256)
extractSubSections(img,numChunks=12,hSplits=3,vSplits=4)