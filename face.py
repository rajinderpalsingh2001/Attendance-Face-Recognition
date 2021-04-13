import numpy as np
import cv2
import face_recognition
import os
import datetime
import pandas as pd

# Image for store image in form of np array
IMAGES = []
# Name For store Name of persion which is from image name
NAME = []


# Read File from the given directory or folder
listOfImageLoation = os.listdir('train')

# For loop for extract single single image from the folder
for name in listOfImageLoation:
    # to extract the name from image like ram.jpg it will give us ram
    NAME.append(name.split('.')[0]) #append name of person from image name
    image = cv2.imread(f"./train/{name}")  #imread() will convert image to array
    IMAGES.append(image)    #append array of image into IMAGES list


def createEncodeImage(images):
    #images -> will encode the image array
    #convert into cascade image, i.e machine readable form

    encodeImages = []   #for storing encoded version of image_array
    for img in images:

        #color coding must be same, otherwise may result in unexpected errors
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #since, at a time single image comes, so we need to add [0]
        #this will recognize face and return a list of tuple having face location coordinates
        encode = face_recognition.face_encodings(img)[0]

        #only encode face part, hence increase processing speed
        #since there's no need of encoding all the image, because recognition occurs with the help of face
        encodeImages.append(encode)

    #at end, encodeImages[] will contain encoded version of faces, in list    
    return encodeImages

def markAttend(name):
    dt=str(datetime.datetime.now())
    date=dt[:10]
    time=dt[11:].split('.')[0]
    
    attendee=[
        [name,date,time]
    ]
    df=pd.DataFrame(attendee,columns=['Name','Date','Time'])
    try:
        df=pd.read_csv('attendence.csv')
    except:
        edf=[]
        edf=pd.DataFrame(attendee,columns=['Name','Date','Time'])
        edf.to_csv('attendence.csv',index=False)
    
    df=pd.read_csv('attendence.csv')
    if(df[(df['Name']==name) & (df['Date']==date)].empty):
        df.to_csv('attendence.csv',index=False,mode='a',header=False)
        print("Attendence marked")
    else:
        print("already ,marked")
    

#encode the IMAGES[], containing all images_array
encodeImageList = createEncodeImage(IMAGES)

print("Start Video Capture")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    '''
        reducing the size of image, to increase the processing speed
        we will use, this image only for processing
    '''
    imgs = cv2.resize(src=img, dsize=(0, 0), fx=0.25, fy=0.25)
    '''
        fx,fy=0.25 means reduce by 25% of it's original size
    '''
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    #--find location of face, from video image--
    '''
        there may be a multiple persons in a image (video Frame)
        so don't add [0], because we need locations of all faces
    '''
    faceLocFrame = face_recognition.face_locations(imgs)

    #encoding images
    '''
        encode image imgs only face portion 
        encodeFrame = face_recognition.face_encodings(image_to_be_encoded, portion_to_be_encoded)
    '''
    encodeFrame = face_recognition.face_encodings(imgs, faceLocFrame)

    #functioning of zip()
    '''
        a=[1,2,3,4,5]
        b=[5,8,3,5,1]
        c=zip(a,b)

        c -> will contain a zipped object
        convert to list to read

        c=list(c)

        #output
        [
            (1,5),
            (2,8),
            (3,3),
            (4,5),
            (5,1)
        ]

        #working
        a= [ 1 , 2 , 3 , 4 , 5 ]
             |   |   |   |   |            
             |   |   |   |   |
        b= [ 5 , 8 , 3 , 5 , 1 ]

        for valueof_a,valueof_b in zip(a,b):
            print(valueof_a,end=',')
                # 1,2,3,4,5,
            print(valueof_b)
                #5,8,3,5,1,
    '''

    #zip(faceLocFrame, encodeFrame) so that, face location and encoded version gets stored at same position
    for faceLoc, encode in zip(faceLocFrame, encodeFrame):

        #from list of encoded faces match the encode version of face
        matches = face_recognition.compare_faces(encodeImageList, encode)
        #returns True, if matches

        #also find distance
        distance = face_recognition.face_distance(encodeImageList, encode)
        '''
            if we find minimum distance, 
            from distance_array then we can find the Name of attendee
            because they are stored at same index in different lists
        '''
        match = np.argmin(distance) #return a index of minimum value

        #if condition only works if 'True'
        if matches[match]:
            markAttend(NAME[match]) #mark the attendence
            y1, x2, y2, x1 = faceLoc    #assign coordinates of face

            '''
                since we reduced the size of image for processing,
                so we need the set the pixels accordingly to draw rectnagle at correct position

                so we need to modify pixels accordingly
            '''
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            #draw rectangle
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)

            #show Name
            cv2.putText(img,NAME[match],(x1,y2+30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)


    cv2.imshow("Frame", img)

    k = cv2.waitKey(33)
    if k == 27:
        break
    elif k == 13:
        cv2.imwrite(f"./capImage/img{np.random.randint(0, 1000)}.png", img)
        continue
    else:
        # print(k)
        pass
