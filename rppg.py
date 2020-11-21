from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtMultimedia as qtm
from PyQt5.QtCore import pyqtSlot 
from numpy import ndarray
import numpy as np 
import time 
import math
#Class rppg is a singleton 
class rppg():
    __instance = None 
    @staticmethod
    def rppgInstance():
        if rppg.__instance == None:
            rppg()
        return rppg.__instance
    
    def __init__(self):
        if rppg.__instance !=None:
            raise Exception("rppg Class is a singleton class")
        else:
            rppg.__instance = self
            self.image = None 
            self.U = [] 
            self.sigmas = []
            self.pulse = []
            self.curr_U = [] 
            self.prev_U = [] 
            self.curr_sigmas = [] 
            self.prev_sigmas = []
            self.signal_sr_b = []
            
    
    @pyqtSlot(ndarray)
    def preImage(self, ndarray):
        None
    
    #############TODO:Implement S2R function####################
    #@pyqtSlot(ndarray)
    def s2r(self, imagedata,face_detect_rate):
        None

      
        if face_detect_rate < 10:
            print("######################")
        
        image = np.array(imagedata)
        row, col, channel = imagedata.shape 
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        skin_mask = generate_skinmap(image)
        r_masked = r[skin_mask==1]
        g_masked = g[skin_mask==1]
        b_masked = b[skin_mask==1]
        values = np.array([r_masked, g_masked, b_masked])
        
        #spatial RGB correlation 
        C = np.matmul(values,values.T) / (row*col)
    
        D,V = np.linalg.eigh(C)
        print("V++")
        print(V)
        diag_ele = D
        #U_, S_, V_ = np.linalg.svd(C)
        sort_index = np.argsort(diag_ele)[::-1]
        sort_diag = sorted(diag_ele)[::-1]
        V = V[:,sort_index]
        print("V---")
        print(V)
        #self.U.append(V)
        #print(self.U)
        #self.sigmas.append(sort_diag)
        
        if face_detect_rate < 10:
            self.curr_U = V 
            self.prev_U = V 
            self.curr_sigmas = sort_diag
            self.prev_sigmas = sort_diag        
        self.prev_U = self.curr_U
        self.curr_U = V 
        self.prev_sigmas = self.curr_sigmas
        self.curr_sigmas = sort_diag
        #print(self.curr_U)
        #print(self.curr_sigmas)
        if face_detect_rate > 10:
            rot = [np.matmul(self.curr_U[:,0].T, self.prev_U[:,1]), np.matmul(self.curr_U[:,0], self.prev_U[:,2])]
            #print(rot)
            print(self.curr_sigmas[0]/self.prev_sigmas[1],self.curr_sigmas[0]/self.prev_sigmas[2] )
            scale = [math.sqrt(self.curr_sigmas[0]/self.prev_sigmas[1]), math.sqrt(self.curr_sigmas[0]/self.prev_sigmas[2])]
            sr = np.array(scale) * np.array(rot)
            sr_bp = np.matmul(sr, [self.prev_U[:,1], self.prev_U[:,2]])
            self.signal_sr_b.append(sr_bp)
            #print(self.signal_sr_b)

        print("s2r")
       
    
    
        

    #############TODO:Implement chrom function####################
    @pyqtSlot(ndarray)
    def chrom(self, imagedata):
        print("Chrom")
        None



def rgb2ycbcr(im):

    im = np.array(im, dtype=int)
    xform = np.array([[.2568, .5041, .0979], [-.1482, -.291, .4392], [.4392, -.3678, -.0714]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr[:,:,0] += 16
    return np.uint8(ycbcr)

def generate_skinmap(img):
    
    height, width = img.shape[:-1]
    output = np.zeros([height, width])
    img_ycbcr = rgb2ycbcr(img)
    cb = img_ycbcr[:,:,1]
    cr = img_ycbcr[:,:,2]
    r,c = np.where((cb>=98) & (cb<=142) & (cr>=133) & (cr<=177))
    for i in range(len(r)):
        output[r[i], c[i]] = 1

    return output
