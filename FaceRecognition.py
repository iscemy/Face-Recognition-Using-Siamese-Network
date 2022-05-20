from cmath import inf
from xml.dom import minidom
from rsa import verify
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
import os
import random
import pickle

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))
        
        vgg16 = models.vgg16(pretrained=True)

        for param in vgg16.features[:24]:
            param.require_grad = False

        num_features = vgg16.classifier[0].in_features

        additional = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512,512),
            nn.ReLU(inplace=True)
        )
        vgg16.classifier = nn.Sequential(*additional) 
        # print(vgg16)

        self.vgg16 = vgg16

        

    def forward_once(self, x):
        #output = self.cnn1(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc1(output)
        output = self.vgg16(x)
        return output

    def forward(self, anchor, neg, pos):
        anchor = self.forward_once(anchor)
        neg = self.forward_once(neg)
        pos = self.forward_once(pos)
        return anchor, neg, pos

class FaceRecognizer():
    def __init__(self, modelPath = "lfw_pre_more_triplet9_512-512-512-vgg-16-last-block_mining_weights", faceDBPath = "faces.db") -> None:
        self.net = SiameseNetwork()
        state = torch.load(modelPath, map_location=torch.device('cpu'))
        self.net.load_state_dict(state)
        self.net.eval()
        self.treshold = 8.5
        self.transformation = transforms.Compose([transforms.Resize((244,244)),
                                     transforms.ToTensor()
                                    ])
        self.facedb = {}
        self.faceDBPath = faceDBPath
        try:
            file = open(faceDBPath, 'rb')
        except FileNotFoundError:
            file = open(faceDBPath, 'wb')
            pickle.dump(self.facedb, file)
            file.close
        try:
            file = open(faceDBPath, 'rb')
            self.facedb = pickle.load(file)
        except EOFError:
            pass
        file.close()
    def __getFeaturesOfFace(self,face):
        f = self.transformation(face)[None, :]
        with torch.no_grad():
            output = self.net.forward_once(f)
        return output
                          
    def compareFaces(self, face1, face2):
        output1 = self.__getFeaturesOfFace(face1)
        output2 = self.__getFeaturesOfFace(face2)
    
        distance = F.pairwise_distance(output1, output2) 
        print(distance)
        if(distance < self.treshold): return True
        else: return False

    def compareFacesWDistance(self, face1, face2):
        output1 = self.__getFeaturesOfFace(face1)
        output2 = self.__getFeaturesOfFace(face2)
    
        distance = F.pairwise_distance(output1, output2) 
        print(distance)
        if(distance < self.treshold): return True, distance.item()
        else: return False, distance.item()

    def __compareFaceWithFeatures(self, face, features):
        faceFeatures = self.__getFeaturesOfFace(face)
        distance = F.pairwise_distance(features, faceFeatures) 
        if(distance < self.treshold): return True
        else: return False
    
    def __compareFeaturesWithFeatures(self, feature1, feature2):
        distance = F.pairwise_distance(feature1, feature2) 
        if(distance < self.treshold): return True, distance
        else: return False, distance

    def __saveChangesInFaceDB(self):
        f = open(self.faceDBPath, "wb")
        pickle.dump(self.facedb, f)
        f.close()

    def verifyFaceWithKnownFace(self, face, faceid):
        if(self.facedb.get(faceid) != None):
            return self.__compareFaceWithFeatures(face, self.facedb[faceid])
        else: return False

    def identifyWithKnownFaces(self, faceq, default = None):
        numOfMatches = 0
        queryFeatures = self.__getFeaturesOfFace(faceq)
        found = False
        min_dist = inf
        for faceid in self.facedb:
            for feature in self.facedb[faceid]:
                r, d = self.__compareFeaturesWithFeatures(queryFeatures, feature)
                d = d.item()
                found  = found or r
                if(d < min_dist): min_dist = d
            if(found): return faceid, min_dist
        return None, min_dist

    def addToKnownFaces(self, faceid, faceImages):
        features = self.facedb.get(faceid)
        if(features == None):
            self.facedb[faceid] = []
        for face in faceImages:
            self.facedb[faceid].append(self.__getFeaturesOfFace(face))
        self.__saveChangesInFaceDB()
        
# fr = FaceRecognizer()
