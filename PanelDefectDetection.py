##<TODO>: Remove unnecessary modeuls from below.
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.io import imread,imsave
import skimage.filters
from skimage import exposure
from skimage import img_as_float
import skimage.morphology
import skimage.segmentation
import skimage.transform
import PIL as Ima
import time
import shutil
import os
import math


def processAllImages(path):
    #Implement reading files from provided path.
    imgList = os.listdir(path=path)
    for imgFile in imgList:
        filePath = path+"\\"+imgFile
        image = cv.imread(filePath,)
        grayImage = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
        processImage(grayImage)


    
def plotImageHistogram(img,bins):
    axes = np.zeros((1,2),dtype=np.object)
    axes[0,0]=plt.subplot(2,1,1)
    imgAxes = axes[0,0]
    axes[0,1] = plt.subplot(2,1,2)
    histAxes = axes[0,1]
    cdfAxes = histAxes.twinx()
    image = img_as_float(image=img)
    imgAxes.imshow(img,cmap=plt.cm.get_cmap(name="gray"))
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
    ##When plotting number of images to be plotted in rows.
    

    currChunk = 0 
    subImgList = []
    startH = startW = 0
    row = col = 1
    # img.append(image)
    for hSplit in range(0,hSplits):
        for vSplit in range(0,vSplits):
            subImageDict = {}
            endH = (hSplit + 1) * chunkHeight
            endW = (vSplit + 1) * chunkWidth
            endHOverlap = endH + hOverlap
            endWOverlap = endW + wOverlap
            
            subImageDict["imgNo"] = currChunk
            subImageDict["startH"]=startH
            subImageDict["endH"]=endHOverlap
            subImageDict["startW"] = startW
            subImageDict["endW"] = endHOverlap 
            subImageDict["row"] = row
            subImageDict["col"] = col
            subImageDict["subImage"] = image[startH:endHOverlap,startW:endWOverlap]
            subImgList.append(subImageDict)    

            #Should we be doing endW - wOverlap so that it starts from where previous column ended.
            col = col + 1
            currChunk = currChunk + 1
            startW = endW - wOverlap
        ## Horizontally we should move to next row.
        ## Vertically reset it 0
        startH = endH - hOverlap
        row = row + 1
        col = 1
        startW = 0

    return subImgList    


def plotImages(images,hSplits =3,vSplits = 4):
    fig, axes = plt.subplots(hSplits, vSplits, figsize=(5, 5))
    if hSplits * vSplits > 1:
        ax = axes.ravel()
        for i in range(0,len(images)):
            ax[i].imshow(images[i]["subImage"],cmap=plt.cm.get_cmap(name="gray"))
    else:
        plt.imshow(images,cmap='gray', vmin = 0, vmax = 255)
    plt.show()


def padImages(images,hSplits = 3, vSplits = 4, vPad=25,hPad=25):
    value = [255,200,120]
    for img in images:
        row = img['row']
        col = img['col']
        ##Top Left Corner
        if (row == 1 and col == 1):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=0,left=0,bottom=hPad,right=vPad,borderType=cv.BORDER_CONSTANT,value=value)
       
        ##Top Row (Excludign Edges)
        if (row == 1 and col> 1 and col < vSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=0,left=vPad,bottom=hPad,right=vPad,borderType=cv.BORDER_CONSTANT,value=value)
       

        #Top Right Image
        if (row ==1 and col == vSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=0,left=vPad,bottom=hPad,right=0,borderType=cv.BORDER_CONSTANT,value=value)


        #First Column excluding edges 
        if (col == 1 and row >1 and row < hSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=hPad,left=0,bottom=hPad,right=vPad,borderType=cv.BORDER_CONSTANT,value=value)
       
                
        #Bottom Left Corner
        if (col == 1 and row == hSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=hPad,left=0,bottom=0,right=vPad,borderType=cv.BORDER_CONSTANT,value=value)
       

        #Right most column (excluding edges)
        if (col == vSplits and row > 1 and row < hSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=hPad,left=vPad,bottom=hPad,right=0,borderType=cv.BORDER_CONSTANT,value=value)
       
        
        ##Bottom row excluding edges
        if (row == hSplits and col > 1 and col < vSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=hPad,left=vPad,bottom=0,right=vPad,borderType=cv.BORDER_CONSTANT,value=value)
        
        ##Bottom Right Corner
        if (row == hSplits and col == vSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=hPad,left=vPad,bottom=0,right=0,borderType=cv.BORDER_CONSTANT,value=value)    

        ##Middle Cells
        if (row > 1 and row < hSplits and col > 1 and col < vSplits):
            img['subImage'] = cv.copyMakeBorder(img['subImage'],top=hPad,left=vPad,bottom=hPad,right=vPad,borderType=cv.BORDER_CONSTANT,value=value)

    return images
def transformImage(img, transformType):
    pass

def segmentImage(img,segmentType):
    pass

def removeBorders(img,hPixels=10,vPixels=10,dumbMethod=1):
    if dumbMethod != 1:
        rows = np.where(np.max(img,0) > 100)[0]
        if rows.size:
            cols = np.where(np.max(img,1) > 100)[0]
            if cols.size:
                startRow = rows[0]
                endRow = rows[-1]+1
                startCol = cols[0]
                endCol = cols[-1]+1
                finalImage = img[startCol:endCol,startRow:endRow]
            else:
                finalImage = img[:1,:1]
        else:
            finalImage = img[:1,:1]
    else:
        height,width = img.shape
        startX = vPixels
        startY = hPixels
        imgWidth = width - startX
        imgHeight = height - startY
        finalImage = img[startY:imgHeight,startX:imgWidth]

    return finalImage
        

def processImage(img):
    img = removeBorders(img,hPixels=50,vPixels=75,dumbMethod=1)
    plotImages(img,hSplits=1,vSplits=1)
    subImages = extractSubSections(img,numChunks=12,hSplits=3,vSplits=4)
    subImages = padImages(subImages,hSplits=3,vSplits=4,vPad=50,hPad=50)
    plotImages(subImages,hSplits=3, vSplits=4)
    

path = "C:\\DeepLearning\\Vision\\EL\\allimg\\test"
processAllImages(path)