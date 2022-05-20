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
        print(vgg16)

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

class LoadTriplets(Dataset):
  def __init__(self, path, transform=None):
    super(LoadTriplets, self).__init__()
    self.casiaImages = {}
    self.image_size = 0
    self.triplets = []
    self.transform = transform
    for identity in tqdm(os.listdir(path)):
      self.casiaImages[identity] = []
      class_path = path + "/" + identity
      for image in os.listdir(class_path):
        if(image[-6:] != "fa.ppm" and image[-6:] != "fb.ppm"): continue
        self.casiaImages[identity].append(class_path + "/" + image)
        self.image_size += 1
        

    positive_pair_size = self.image_size/4
    negative_pair_size = self.image_size/4
    print(f"image size {self.image_size}")
    for tries in range(10):
      for person1 in self.casiaImages:
        person1Images = self.casiaImages[person1]
        if(len(person1Images) >= 2):
          triplet = (person1Images.pop(0), person1Images.pop(0), 0)
          for person2 in self.casiaImages:
            person2Images = self.casiaImages[person2]
            if (person1 == person2) or (len(person2Images) < 1): continue
            triplet = (triplet[0], triplet[1], person2Images.pop(0))
            self.triplets.append(triplet)
            break
    
  def __len__(self):
        return len(self.triplets)
  def __getitem__(self,index):

    img0 = Image.open(self.triplets[index][0]).convert("RGB")
    img1 = Image.open(self.triplets[index][1]).convert("RGB")
    img2 = Image.open(self.triplets[index][2]).convert("RGB")

    if(self.transform != None):
      img0 = self.transform(img0)
      img1 = self.transform(img1)
      img2 = self.transform(img2)
    
    return img0, img1, img2

transformation = transforms.Compose([transforms.Resize((244,244)),
                                     transforms.ToTensor()
                                    ])

net = SiameseNetwork()
state = torch.load("lfw_pre_more_triplet9_512-512-512-vgg-16-last-block_mining_weights", map_location=torch.device('cpu'))
net.load_state_dict(state)
net.eval()

feretColorDataset = LoadTriplets("feret_ds/")

false_negative = 0
false_positive = 0
true_negative = 0
true_positive = 0
i = 1000

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  
    
def imsave(img,text=None, filename = ""):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10}) 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("false/"+filename + ".png")
    plt.close() 

i = 0
for a_f, p_f, n_f in feretColorDataset:
  i += 1
  # if i < 0: break
  a = transformation(a_f)[None, :]
  p = transformation(p_f)[None, :]
  n = transformation(n_f)[None, :]
  with torch.no_grad():
    output1, output2, output3 = net(a, p, n)
    positive_dist = F.pairwise_distance(output1, output2) # between anchor and positive 
    negative_dist = F.pairwise_distance(output2, output3) # negative and positive
  p_score = positive_dist.item()
  n_score = negative_dist.item()
  
  treshold = 8

  if(p_score < treshold):
    true_positive += 1
    # if(p_score < treshold and p_score > treshold - 1):
    #   imsave(torchvision.utils.make_grid(torch.cat((a, p), 0)), "true positive close to edge " + str(p_score), str(i) + "_tpb")
    # if(p_score < treshold/2):
    #   imsave(torchvision.utils.make_grid(torch.cat((a, p), 0)), "true positive good match " + str(p_score), str(i) + "_tpg")
    if(p_score < treshold-2 and p_score > treshold - 4.5):
      imsave(torchvision.utils.make_grid(torch.cat((a, p), 0)), "im " + str(p_score), str(i) + "_tib")
  else:
    false_negative += 1
    imsave(torchvision.utils.make_grid(torch.cat((a, p), 0)), "false negative " + str(p_score), str(i) + "_fn")
    # print(n_score)
  if(n_score > treshold):
    true_negative += 1
  else:
    false_positive += 1
    imsave(torchvision.utils.make_grid(torch.cat((n, p), 0)), "false positive " + str(n_score), str(i) + "_fp")
    # print(n_score)
  #print(positive_dist, negative_dist)

print(f"true_positive {true_positive} true_negative {true_negative} false_negative {false_negative} false_positive {false_positive}")
total = true_positive + true_negative + false_positive  + false_negative
print(total)
print((true_positive + true_negative) / total)
  