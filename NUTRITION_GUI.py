# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:13:58 2018

@author: prasadi
"""
#import the necessary modules
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2
import numpy as np
import time
from tkinter import messagebox

root=tk.Tk()
root.geometry("1500x800")
root.title("Nutrition Level Detection")
root.configure(background='aquamarine')
bgImg=cv2.imread("GUI_Background.jpg")
bgImg=cv2.cvtColor(bgImg, cv2.COLOR_BGR2RGB)
bgImg = Image.fromarray(bgImg)
bgImg = ImageTk.PhotoImage(bgImg)
bglbl = tk.Label(image=bgImg)
bglbl.image = bgImg
bglbl.place(x=0, y=0, height=800, width=1500)

L1=tk.Label(root,font=('arial',30,'bold'),text="Plant Nutrition Level Detection System",fg="brown4",bg='bisque2',bd=10)
L1.place(x=300, y=10, height=60, width=800)
localtime=time.asctime(time.localtime(time.time()))
lblInfo2 =tk.Label(root, font=('arial',20,'bold'),text=localtime,fg="brown1",bg='bisque2',bd=10)
lblInfo2.place(x=300, y=60, height=60, width=800)


def loadClicked():
    global panelA, panelB

    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        img1=cv2.imread(path)     #read the image
        img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  #convert BGR image to RGB image
        #resize any image into a standard size (300x300 pixels)
        img2=cv2.resize(img1, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
        
        #grabcut image--> to extract only the focused image
        mask=np.zeros(img2.shape[:2],np.uint8)
        bgdModel=np.zeros((1,65),np.float64)
        fgdModel=np.zeros((1,65),np.float64)
        rect=(1,1,img2.shape[1]-2,img2.shape[0]-2)
        cv2.grabCut(img2,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img=img2*mask2[:,:,np.newaxis]
        
        #convert the leaf to HSV color model--> to get the average intesity of the image
        hsv= cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        V=hsv[:,:,2]     #extract only the 'Value' array from 3D array--> Value represents the intensity(0,255)
        V1=np.reshape(V,(V.shape[0]*V.shape[1],1))  #reshape to a 1D array
        non_black_inv=np.where(V1!=0)[0]    #get the indexes having any color other than black(v=0 -->black color)
        if(len(non_black_inv)!=0): 
            inverse=np.average(V1[non_black_inv])  #take the average of intensity values
            
            #function defined to set the intensity values in a linear curve
            def adjust_gamma(image, invgamma=1.0):
                invGamma = 1.0 * invgamma
                table = np.array([((i / 255.0) ** invGamma) * 255
                   for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(image, table)
            invgamma = float(inverse/100.0)                
            img3 = adjust_gamma(img, invgamma=invgamma) #call the adjust_gamma function to adjust the intensity values of the image
            
            #green color removal code
            hsv = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV) #convert the intensity adjusted image to HSV color model
            mask = cv2.inRange(hsv, (36, 0, 0), (80, 255,255)) #mask of green (36,0,0) ~ (80, 255,255) 
            imask = mask>0  #slice the green
            green = np.zeros_like(img3, np.uint8)
            green[imask] = img3[imask]  #green contains only the green(healthy) parts of the leaf
            img4=img3-green   #subtract green part from the whole leaf-->img4 contains only the unhealthy parts of the leaf
            
            sum1=np.array(img).sum(axis=2)
            sum2=np.reshape(sum1,(sum1.shape[0]*sum1.shape[1],1))
            sum3=np.where(sum2!=0)[0]
            sum4=len(sum3)    #number of pixels in img3(whole leaf with both healthy and unhealthy parts)
            
            #Analyze the unhealthy part of the leaf
            hsv=cv2.cvtColor(img4, cv2.COLOR_RGB2HSV_FULL)  #convert to HSV Full model
            h=hsv[:,:,0]
            v=hsv[:,:,2]        
            h2=np.reshape(h,(h.shape[0]*h.shape[1],1))
            v2=np.reshape(v,(v.shape[0]*v.shape[1],1))
            non_black=np.where(v2!=0)[0]   #take only non black indexed of pixels
            addition=np.sum(h2[non_black])  #add the H values
            num_non_black=len(non_black)  #number of non black pixels
            percent=(num_non_black/sum4)*100  #percentage between healthy and unhealthy parts of the leaf
            if(percent>3):   
                avg=addition/num_non_black  #average H value of the unhealthy part of the leaf(H-->Hue represents the color)
                if(avg>0 and avg<50):   #if the average is between (0,80)--> yellow color-->Nitrogen or Potassium
                    if (percent>55):   #if the percentage is greater than 55--> Nitrogen
                        typeN='Nitrogen'
                 
                    else:
                        typeN='Potassium' #if the percentage is lower than 55--> Potassium
                              
                elif(avg>50 and avg<360):
                    typeN='Phosphorus'   #else it is Phosphorus  
                           
            else:  #else it is a healthy leaf
                avg=0
                typeN='Healthy'
                percent=0
            
        else:    #if the image is not very close image there will be no remaining part after grabcut
            img4=img
            messagebox.showerror("Error", "Take the picture again!")
            typeN='--'
            percent=0


        lbltype=tk.Label(root, font=('arial',15,'bold'),text="Type of Low Nutrition:",fg="IndianRed4",bg='bisque2',bd=10)
        lbltype.place(x=350, y=200, height=60, width=400)
        lblpercent =tk.Label(root, font=('arial',15,'bold'),text="Low Nutrition level",fg="IndianRed4",bg='bisque2',bd=10)
        lblpercent.place(x=350, y=250, height=60, width=400)
        lbltype_val=tk.Label(root, font=('arial',15,'bold'),text=typeN,fg="red4",bg='bisque2',bd=10)
        lbltype_val.place(x=650, y=200, height=60, width=400)
        lblpercent_val =tk.Label(root, font=('arial',15,'bold'),text=str(round(percent))+"%",fg="red4",bg='bisque2',bd=10)
        lblpercent_val.place(x=650, y=250, height=60, width=400)    
              
        # convert the images to PIL format...
        image2 = Image.fromarray(img2)
        #image = Image.fromarray(img)
        image4 = Image.fromarray(img4)
        
        # ...and then to ImageTk format
        image2 = ImageTk.PhotoImage(image2)
        image4 = ImageTk.PhotoImage(image4)

        # if the panels are None, initialize them
        if panelA is None or panelB is None :
            # the first panel will store our original image
            panelA = tk.Label(image=image2,bg='papaya whip')
            panelA.image = image2
            panelA.place(x=350, y=320, height=350, width=350)
            # while the second panel will store the green removal
            panelB = tk.Label(image=image4,bg='papaya whip')
            panelB.image = image4
            panelB.place(x=700, y=320, height=350, width=350)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image2)
            panelB.configure(image=image4)
            panelA.image = image2
            panelB.image=image4

#==================================================Info====================================================
panelA = None
panelB = None

btnload=tk.Button(root,padx=8,pady=8,bd=8,fg="red4",font=('arial',12,'bold'),
                  text="Load Image",bg="salmon1", command=loadClicked).place(x=650, y=130, height=50, width=110)

root.mainloop()