import numpy as np
import cv2,time
import pickle as pkl
import os
####################################
#Text parameters
font=cv2.FONT_HERSHEY_SIMPLEX
font_scale=0.7
color=(0,255,255)#white
font_thickness=2
line_type=cv2.LINE_AA
#####################################
labels = {}
Face_Classifier=cv2.CascadeClassifier('/media/nile/KINGSTON/loma/Data/haarcascade_frontalface_alt2.xml') #we are using haar cascade classsifier
recognizer=cv2.face.createLBPHFaceRecognizer()

recognizer.load("trainner.yml")
##########################################
#time variables
capture_duration = 20
start_time = time.time()
###########################################
with open("labels.pikl",'rb') as f:
    labels=pkl.load(f)    #load  label ids in file labels.pikl
    labels_inverted={k:v for v,k in labels.items()}
    
right_guess_count=0
unknown_count=1
action=0

img_counter=0
cam = cv2.VideoCapture(0)            #start capturing

while cam.isOpened() and ( int(time.time() - start_time) < capture_duration ):
    check, frame = cam.read()        #start taking frames
    if check==True:
        
        Flibaya=cv2.flip(frame,1)       #flip the frame
        grey=cv2.cvtColor(Flibaya,cv2.COLOR_BGR2GRAY)              #haar works and uses gray image 
        Face_detect=Face_Classifier.detectMultiScale(grey,1.1,4)     #specify a rectangle around the face

        for(x,y,w,h) in Face_detect:
            roi_gray_flipped=grey[y:y+h,x:x+w]
            cv2.rectangle(Flibaya,(x,y),(x+w,y+h),(0,0,255),2)    #Give the rectangle the dimentions of the detected fac
            id_, conf = recognizer.predict(roi_gray_flipped)
            if conf >=65:
                #print(id_,conf)
                print(labels_inverted[id_])  #printing names
                name=labels_inverted[id_]
                right_guess_count+=1
            else:
                 name="unknown person"
                 unknown_count+=1

            name2=str(conf)
            cv2.putText(Flibaya,name,(x,y),font,font_scale,color,font_thickness,line_type)
            cv2.putText(Flibaya,name2,(x,y+h),font,font_scale,color,font_thickness,line_type)      
                
        cv2.imshow('my webcam', Flibaya)                        #Start shoing the camera
    #if k == 27:               # if esc is pressed it will quit
     #   break  
    elif k %256 == 32:           #space pressed,take an image
        cropped=Flibaya[y:y+h,x:x+w]
        print ("Cropped is{} ".format(cropped))
        img_name = "/home/nu/loma/algorith pics/opencv_frame_{}.png".format(img_counter) #opencv_frame_{1}.png
        cv2.imwrite(img_name, Flibaya)
        print("{} written!".format(img_name))
        img_counter += 1
    else:
        break
    k = cv2.waitKey(1)                                   #Accepts key interface    

correctness_perc=(right_guess_count/(right_guess_count+unknown_count))*100

if correctness_perc >=70:
    action=1
else:
    action=0

print(action)
print(correctness_perc)
cv2.destroyAllWindows() #after an action is taken(esc or space) window will collapse





 
