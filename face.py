from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np



def face_detector():
    faceDetect = cv2.CascadeClassifier('C:/Users/hp/Desktop/facerecognistaion/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/hp/Desktop/facerecognistaion/data/haarcascade_eye.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:/Users/hp/Desktop/facerecognistaion/recognizer/trainner.yml')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    id=0

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)




    while(True):
        ret, img =cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            # Detects eyes of different sizes in the input image
            eyes = eye_cascade.detectMultiScale(roi_gray)

            #To draw a rectangle in eyes
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
            id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence < 60):
                confidence = "  {0}%".format(round(100 - confidence))
                if(id==1):
                    id="vikas"
                elif(id==2):
                    id="Harsh"
                elif(id==3):
                    id="Akash"
  
            else:
                id="unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            
        cv2.imshow('camera',img)
    
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    

def face_creater():
    cam = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier('C:/Users/hp/Desktop/facerecognistaion/data/haarcascade_frontalface_default.xml')

    Id=input('enter your id')
    sampleNum=0
    while(True):                                        
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
            #incrementing sample number 
            sampleNum=sampleNum+1
            #saving the captured face in the dataset folder
            cv2.imwrite("C:/Users/hp/Desktop/facerecognistaion/DATASETS/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('frame',img)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 20
        elif sampleNum>20:
            break
    cam.release()
    cv2.destroyAllWindows()







    def train():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path='C:/Users/hp/Desktop/facerecognistaion/DATASETS'
        detector= cv2.CascadeClassifier('C:/Users/hp/Desktop/facerecognistaion/data/haarcascade_frontalface_default.xml')

        def getImagesWithID(path):
            #get the path of all the files in the folder
            imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
            #print(imagePaths)
            #getImagesWithID(path)
            #create empth face list
            faces=[]
            #create empty ID list
            IDs=[]
            #now looping through all the image paths and loading the Ids and the images
            for imagePath in imagePaths:
                #loading the image and converting it to gray scale
                faceImg=Image.open(imagePath).convert('L')
                #Now we are converting the PIL image into numpy array
                faceNp=np.array(faceImg,'uint8')
                #getting the Id from the image
                ID=int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                print(ID)
                IDs.append(ID)
                cv2.imshow("training",faceNp)
                cv2.waitKey(10)
            return IDs, faces
        
            
            
    


        Ids,faces = getImagesWithID(path)
        recognizer.train(faces, np.array(Ids))
        recognizer.write('C:/Users/hp/Desktop/facerecognistaion/recognizer/trainner.yml')

        cv2.destroyAllWindows()

    train()

    
#Gui code    

SET_WIDTH=700
SET_HEIGHT=400   

root = tk.Tk()
root.title("True_face")



cv_img=cv2.cvtColor(cv2.imread("C:/Users/hp/Desktop/img/favicon1.png"),cv2.COLOR_BGR2RGB)

canvas=tk.Canvas(root,width=SET_WIDTH,height=SET_HEIGHT)

#im = PIL.ImageTK.PhotoImage(image=PIL.Image.fromarray(cv2_Img))

#image_on_canvas=canvas.create_image(0,0,ancho=tk.NW,image=im)

canvas.pack()




background_image=tk.PhotoImage(file = r"C:\Users\hp\Desktop\img\bg.jpg")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


#root.configure(background='black')





#add img in buttons
photo = PhotoImage(file = r"C:\Users\hp\Desktop\img\webcam.png") 
photoimage = photo.subsample(17, 20)

Button(root,
text = "Detector",font=("Bradley Hand ITC",20),image = photoimage,compound = TOP,
background = "skyblue",
foreground="white",
activebackground="#FF7F50",
command=face_detector,
bg="skyblue"
).place(x=500,y=20)


#add img in buttons
photo1 = PhotoImage(file = r"C:\Users\hp\Desktop\img\addimg.png") 
photoimage1 = photo1.subsample(10, 3)

Button(root,
text = " Add_Face",font=("Bradley Hand ITC",20),image = photoimage1,compound = RIGHT,
background = "black", 
foreground="green",
activebackground="#FF7F50",
command=face_creater,
bg="orange"
).place(x=300,y=180)


#add img in buttons
photo2 = PhotoImage(file = r"C:\Users\hp\Desktop\img\quit.png") 
photoimage2 = photo2.subsample(3, 3)

Button(root,
text = "quit",font=("Bradley Hand ITC",20),image = photoimage2,compound = LEFT,
background = "red",
foreground="white",
command=root.destroy,
bg="red"

).place(x=100,y=300)

root.mainloop()
    




        
