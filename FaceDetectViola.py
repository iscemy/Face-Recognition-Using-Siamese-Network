from venv import create
from cv2 import imshow
from joblib import dump, load
from matplotlib import pyplot as plt
from scipy.misc import face
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.transform import integral_image
from skimage import io
import numpy as np
import time

from sklearn.preprocessing import scale

##FaceDetectorModel = {"feature_types" : feature_type_sel, "feature_coord" : feature_coord, "classifier" : clf}

class FaceDetector:
    def __init__(self, model = "FaceDetectorModelP2300N6000W24H24.joblib") -> None:
        dict = load(model)
        self.rf = dict["classifier"]
        self.feature_coords = dict["feature_coord"]
        self.feature_types = dict["feature_types"]
        self.cascadeSizeX = dict["cascade_size_x"]
        self.cascadeSizeY = dict["cascade_size_y"]
        ##self.featureCoordsScaled = {1.0 : self.feature_coords}
        self.featureCoordsScaled = {}
        self.lookupTable = {}
        s = 4
        scales = [] 
        print(self.cascadeSizeX, self.cascadeSizeY)
        for i in range(15):
            scales.append(s)
            s = s * 1.10
        self.__createScalesForClassifier(scales)
        
        print("detector ready")

    def __multiplyArrayOfTuplesWithScaler(self, arr, c):
        result = np.empty_like(arr)
        for i, m in enumerate(arr):
            result[i] = []
            for mm in m:
                rr = [( round(mm[0][0] * c), round(mm[0][1] * c)), (round(mm[1][0] * c), round(mm[1][1] * c))]
                result[i].append(rr)

        return result
                


    def __createScalesForClassifier(self,scales):
        #print(np.arange(begin, end, step) * [self.cascadeSizeX, self.cascadeSizeY])
        for s in scales:
            self.featureCoordsScaled[s] = self.__multiplyArrayOfTuplesWithScaler(self.feature_coords, s)

    

    def __extract_feature_image(self, ii, offsetX = 0, offsetY = 0, scale = 1.0):
        return haar_like_feature(ii, offsetY, offsetX, int(self.cascadeSizeX*scale),
                                int(self.cascadeSizeY*scale),
                                feature_type=self.feature_types,
                                feature_coord=self.featureCoordsScaled[(scale)])
    ##ii integral image
    def __applyCascadeAtPos(self, ii, x,  y, scale = 1.0):
        haarFeatures = self.__extract_feature_image(ii, x, y, scale)
        return (self.rf.predict([haarFeatures]))[0]
    
    def detectFaces(self, image):
        ii = integral_image(image)
        faces = []
        
        for cascade in self.featureCoordsScaled:
            start_time = time.time()
            print("starting scale " + str(cascade * self.cascadeSizeX))
            faces += self.applyCascadeToAllPositions(ii,cascade)
            print("takes %s seconds ---" % (time.time() - start_time))

        faces_filtered = []

        
        i = 0
        j = 0
        #filtering positionr
        
        while i < len(faces):
            j = i + 1
            isBestFace = False
            bestFaceScore = 0
            face1 = faces[i]
            while j < len(faces):
                face2 = faces[j]
                dx = face1[0][0] - face2[0][0]
                dy = face1[0][1] - face2[0][1] 
                if (dx * dx + dy * dy < 1152):
                    if face1[2][1] > bestFaceScore:
                        bestFaceScore = face1[2][1]
                        isBestFace = True
                    else:
                        isBestFace = False
                j += 1
            if isBestFace:
                #if face1[2][1] > 0.505:
                faces_filtered.append(face1)
            i += 1

        i = 0
        j = 0


        #return faces
        return faces_filtered


    def applyCascadeToAllPositions(self, ii, scale = 1.0):
        height = ii.shape[0]
        width = ii.shape[1]
        result = []
        
        x_step = round(1 * scale)
        y_step = round(1 * scale)

        i_si = 0
        y = 0
        while (y + scale*self.cascadeSizeY) <= height:
            x = 0
            while (x + scale*self.cascadeSizeX) <= width:
                i_si+=1
                x += x_step
            y += y_step

        scaleFeatures = np.empty((i_si + 1, self.feature_coords.shape[0]))
        scaleFeaturesPos = []
        i_si = 0
        y = 0
        n_nan_lines = 0
        while (y + scale*self.cascadeSizeY) <= height:
            x = 0
            while (x + scale*self.cascadeSizeX) <= width:
                i_si += 1
                scaleFeatures[i_si] = self.__extract_feature_image(ii, x, y, scale)
                if( np.any(np.isnan(scaleFeatures[i_si])) or np.any(np.isinf(scaleFeatures[i_si])) ):
                    ##print(np.isnan(scaleFeatures[i_si]).sum())   # True wherever nan
                    ##print(np.isinf(scaleFeatures[i_si]).sum())     # True wherever pos-inf or neg-inf
                    n_nan_lines += 1
                scaleFeaturesPos.append([int(x), int(y)])
                x += x_step
            y += y_step
        ##print(n_nan_lines)
        ##print(np.isnan(scaleFeatures[i_si]).sum())
        ##print(np.isinf(scaleFeatures[i_si]).sum())

        try:
            predictions = self.rf.predict(scaleFeatures)
            predictions_prob = self.rf.predict_proba(scaleFeatures)
            for xy, p in enumerate(predictions):
                if p == 1:
                    prob = predictions_prob[xy]
                    topleft = tuple(scaleFeaturesPos[xy])
                    bottomright = (scaleFeaturesPos[xy])
                    diff = ([scale * self.cascadeSizeX, scale * self.cascadeSizeY])
                    bottomright[0] = int(bottomright[0] + diff[0])
                    bottomright[1] = int(bottomright[1] + diff[1])
                    print(bottomright)
                    print(prob)
                    result.append([topleft, (bottomright[0], bottomright[1]), prob])
        except:
            pass
        return result

import cv2

import numpy as np
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.data import lfw_subset
import os




def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   

    for image_path in os.listdir(img_folder):
       
      image= cv2.imread(img_folder+image_path, cv2.IMREAD_GRAYSCALE)
      
      try:
        #image=cv2.resize(image, (18, 25),interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        img_data_array.append(image)
        
      except:
        pass


    return img_data_array

if __name__ == "__main__":

    # fd32 = FaceDetector("FaceDetectorModelP100N200W32H32.joblib")
    # fd28 = FaceDetector("FaceDetectorModelP400N800W28H28.joblib")
    # fd24 = FaceDetector("FaceDetectorModelP200N300W24H24.joblib")
    fdf = FaceDetector("FaceDetectorModelP1500N2000W24H24.joblib")
    fd = FaceDetector()
    image= cv2.imread("Alec_Baldwin_0002.jpg", cv2.IMREAD_GRAYSCALE) 

    image=cv2.resize(image, (int(image.shape[1]), int(image.shape[0])),interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')


    result = fd.detectFaces(image)
    for index, r in enumerate(result):
        cv2.rectangle(image, r[0], r[1], (255, 255, 255), 1)
    #fd.detectFaces(im)

    window_name = 'Image'
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

    print("load")



