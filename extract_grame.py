import os
import cv2
import matplotlib.pyplot as plt
import glob
import resnetencoder_gram
#from networks_gram import resnetencoder_gram
from PIL import Image
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn 
def gramx(x, y):
    w1 = x.detach().cpu().numpy().squeeze()
    w2 = y.detach().cpu().numpy().squeeze()
    #w = x.detach().numpy() 
    #w = w.cpu().numpy()
    m = len(w1)  
    n = len(w1[0]) 
    X_train = np.empty((2, 3, m, n))  
    img_ndarrayx = np.resize(w1, (3, m, n))
    img_ndarrayx = torch.Tensor(img_ndarrayx)
    img_ndarrayy = np.resize(w2, (3, m, n))
    img_ndarrayy = torch.Tensor(img_ndarrayy)
    X_train[1] = img_ndarrayx
    X_train[0] = img_ndarrayy
    X_train = torch.Tensor(X_train) 
    b = [i for i in range(0, 2)]
    resnetencoder = resnetencoder_gram.ResnetEncoder(18, False)
    b = resnetencoder(X_train)
    (gram1, gram2) = (b[0].flatten(), b[1].flatten())
    similarity = torch.cosine_similarity(gram1, gram2, dim=0)
    return similarity
