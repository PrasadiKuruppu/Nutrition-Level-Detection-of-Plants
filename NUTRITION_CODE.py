# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:13:40 2018
@author: prasadi
NEUTRITION LEVEL DETECTION
Input--> image
Output--> 1.Average of Hue value
          2.Percentage of the leaf having low neutrition level
          3.Type of low neutrition(Nitrogen,Potassium,Phosphorus)
          4.Time spent to run the code
In this code I have used color of the leaf to detect the type of low 
neutrition.Nitrogen and Potassium have yellow color and Phosphorus has 
a purple color when the leaves are suffering from low neutrition level 
of respective neutrient.          
"""
'''In the code first I have identified the difference between yellow and purple 
color using HSV color model. If the leaf is having purple color then the type will 
get identified as Phosphorus. I have used percentage of yellow colort part of the 
leaf to identify the difference between the types Nitrogen and Potassium. Because 
for low Potassium level the edges of the leaf get the yellow color and then spred 
to the middle areas of the leaf. But for low Nitrogen level whole leaf become into 
yellow color.'''
#import the necessory modules
import time
start_time = time.time()
import numpy as np
from matplotlib import pyplot as plt
import cv2

img1=cv2.imread("N2.jpg")    #read the image
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)   #convert BGR image to RGB image
plt.imshow(img1),plt.show()

#resize any image into a standard size (300x300 pixels)
img2=cv2.resize(img1, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
plt.subplot(1,4,1),plt.imshow(img2)

#grabcut image--> to extract only the focused image
mask=np.zeros(img2.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(1,1,img2.shape[1]-2,img2.shape[0]-2)
cv2.grabCut(img2,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
img=img2*mask2[:,:,np.newaxis]

#convert the leaf to HSV color model--> to get the average intesity of the image
hsv1= cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
V=hsv1[:,:,2]   #extract only the 'Value' array from 3D array--> Value represents the intensity(0,255)
V1=np.reshape(V,(V.shape[0]*V.shape[1],1)) #reshape to a 1D array
non_black_inv=np.where(V1!=0)[0]  #get the indexes having any color other than black(v=0 -->black color)
if(len(non_black_inv)!=0):   
    inverse=np.average(V1[non_black_inv])   #take the average of intensity values
    
    #function defined to set the intensity values in a linear curve
    def adjust_gamma(image, invgamma=1.0):       
       invGamma = 1.0 * invgamma
       table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")    
       return cv2.LUT(image, table)
    
    invgamma = inverse/100                       
    img3 = adjust_gamma(img, invgamma=invgamma)  #call the adjust_gamma function to adjust the intensity values of the image
    
    plt.subplot(1,4,2),plt.imshow(img)
    plt.subplot(1,4,3),plt.imshow(img3)
    
    #green color removal code
    hsv = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV)  #convert the intensity adjusted image to HSV color model
    mask = cv2.inRange(hsv, (36, 0, 0), (80, 255,255))  #mask of green (36,0,0) ~ (80, 255,255)    
    imask = mask>0    #slice the green
    green = np.zeros_like(img3, np.uint8)
    green[imask] = img3[imask]     #green contains only the green(healthy) parts of the leaf
    img4=img3-green   #subtract green part from the whole leaf-->img4 contains only the unhealthy parts of the leaf
    plt.subplot(1,4,4),plt.imshow(img4)
    
    sum1=np.array(img3).sum(axis=2)
    sum2=np.reshape(sum1,(sum1.shape[0]*sum1.shape[1],1))
    sum3=np.where(sum2!=0)[0]
    sum4=len(sum3)   #number of pixels in img3(whole leaf with both healthy and unhealthy parts)
    
    #Analyze the unhealthy part of the leaf
    hsv=cv2.cvtColor(img4, cv2.COLOR_RGB2HSV_FULL) #convert to HSV Full model
    h=hsv[:,:,0]
    v=hsv[:,:,2]
    h2=np.reshape(h,(h.shape[0]*h.shape[1],1))
    v2=np.reshape(v,(v.shape[0]*v.shape[1],1))
    non_black=np.where(v2!=0)[0]  #take only non black indexed of pixels
    aaa=h2[non_black]
    addition=np.sum(h2[non_black])   #add the H values
    num_non_black=len(non_black)  #number of non black pixels
    percent=(num_non_black/sum4)*100   #percentage between healthy and unhealthy parts of the leaf
    if(percent>3):   
        avg=addition/num_non_black  #average H value of the unhealthy part of the leaf(H-->Hue represents the color)
        if(avg>0 and avg<50):   #if the average is between (0,80)--> yellow color-->Nitrogen or Potassium
            if (percent>55):   #if the percentage is greater than 55--> Nitrogen
                typeN='Nitrogen'
         
            else:
                typeN='Potassium' #if the percentage is lower than 55--> Potassium
                      
        elif(avg>50 and avg<255):
            typeN='Phosphorus'   #else it is Phosphorus  
                   
    else:  #else it is a healthy leaf
        avg=0
        typeN='Healthy'
        percent=0
         
    print('Average:',avg)
    print('Percentage:',percent)
    print('Type:',typeN)
    
    

else:  #if the image is not very close image there will be no remaining part after grabcut
    print("Take the picture again!")  
    
print("---Time: %s seconds ---" % (time.time() - start_time))
