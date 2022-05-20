import cv2
import numpy
from PIL import Image
class FaceDetector:
    def __init__(self, haarCascadeFileName = "haarcascade_frontalface_default.xml") -> None:
        self.face_cascade = cv2.CascadeClassifier(haarCascadeFileName)
        

    ##img needs to be gray
    def detectFaces(self, img):
        # Detect the faces
        img = numpy.array(img)
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = self.face_cascade.detectMultiScale(frame_gray)
        faces_m = []
        wf = 1.5
        hf = 2.5
        for (x, y, w, h) in faces:
            x = x - (wf*w - w)/2
            y = y - (hf*h - h)/2
            x = max(int(x),0)
            y = max(int(y),0)
            faces_m.append((x,y,int(w*wf),int(h*hf)))
        # Display
        # cv2.imshow("faces",img)
        # cv2.waitKey(0) 

        return faces_m
    
    def cropFaces(self, img, faces):
        croppedFaces = []
        for (x, y, w, h) in faces:
            cimg = Image.fromarray(numpy.array(img)[int(max(y,0)):y+h, int(max(x,0)):x+w])
            croppedFaces.append(cimg)
            # cv2.imshow("faces",numpy.array(cimg))
            # cv2.waitKey(1) 
        return croppedFaces

    def cropFace2Coords(self, img, faces):
        croppedFaces = []
        for r0, r1, pred in faces:
            x = r0[0]
            y = r0[1]
            w = r1[1] - r0[1]
            h = r1[0] - r0[0]
            print(r0, r1)
            cimg = Image.fromarray(numpy.array(img)[int(max(y,0)):y+h, int(max(x,0)):x+w])
            croppedFaces.append(cimg)
            # cv2.imshow("faces",numpy.array(cimg))
            # cv2.waitKey(1) 
        return croppedFaces        

# image = cv2.imread("1.jpg", cv2.COLOR_BGRA2RGB)
# i = image
# fd = FaceDetector()
# facePositions  = fd.detectFaces(image)
# faces = fd.cropFaces(i, facePositions)


