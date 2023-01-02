import os
from PIL import Image
import numpy as np
import cv2,time
import pickle as pkl #to save label  ids in faces.py


dsize=(700,700)
Face_Classifier=cv2.CascadeClassifier('/media/nile/KINGSTON/loma/Data/haarcascade_frontalface_alt2.xml') #we are using haar cascade classsifier to detect faces
recognizer=cv2.face.createLBPHFaceRecognizer()  #use LBPHFaceRecognizer model""" (facial recogniser)
# gives the current file directory
Base_direc=os.path.dirname(os.path.abspath(__file__)) 
#join the image file path to the current file directory and give me the "images" file path
Image_direc=os.path.join(Base_direc,"images")
print(Image_direc)
current_id=0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(Image_direc):
    for file in files:#search in files that exists in "images" folder
        if file.endswith("png") or file.endswith("jpg"): #search for all png and jpg files
            path=os.path.join(root,file)#add the image path to the root path 
            #print(path)#print all file paths
            #os.path.dirname(path)#shows the path till the folder that contains this path
            #os.path.basename(os.path.dirname(path)) #shows only the folder name containing that file which we call abel
            label=os.path.basename(os.path.dirname(path)).replace(" ","_").lower()
            #print(label,path)
            #giving each folder an ID 
            if label not in label_ids:
                label_ids[label]=current_id
                current_id+=1
                id_=label_ids[label]
            
            #x_train.append(label)
            #x_train.append(path)
            pil_image=Image.open(path).convert("L") #grey scale in pillow
            ########################################
            #Resize images
            img_resized = pil_image.resize(dsize,Image.ANTIALIAS)
            #print(output_resized.shape)
            ########################################
            image_array=np.array(img_resized,"uint8")
            #print(image_array)
            #print(image_array.shape)
            #check if there is an empty frame
            if  image_array is None:
                print("I foound an empty picture {}".format(path))
            
            detect_faces=Face_Classifier.detectMultiScale(image_array,1.1,4) #detects face 
            for(x,y,w,h) in detect_faces:
                ROI=image_array[y:y+h,x:x+w]  #region of interest is the rectangle dimensions around face
                print(ROI)
                x_train.append(ROI)
                y_labels.append(id_)
 #print the dictionary that contains labels and their ids
               
with open("labels.pikl",'wb') as f:
    pkl.dump(label_ids,f)    #save label ids in file labels.pikl
     
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml") #save the recogniser
