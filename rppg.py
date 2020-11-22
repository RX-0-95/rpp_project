from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtMultimedia as qtm
from PyQt5.QtCore import pyqtSlot 
from numpy import ndarray
import numpy as np
from timer import * 
#Class rppg is a singleton 
# part of the code from ##TODO: citation 

MIN_BMP = 50
MAX_BMP = 200

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
            self.timer = timer.timerInstance() 
            self.frame_buffers = [] 
            self.max_buffer_size = 300 
            self.times = [] 
            self.output_dim = 0 
            self.freqs = []
            self.fft = []
            self.bpms = []
            self.bpm = 0
            self.idx = 1

    
    @pyqtSlot(ndarray)
    def preImage(self, ndarray):
        None
    
    #############TODO:Implement S2R function####################
    @pyqtSlot(ndarray)
    def s2r(self, imagedata):
        print("s2r")
        None

    #############TODO:Implement chrom function####################
    @pyqtSlot(ndarray)
    def chrom(self, imagedata):
        print("Chrom")
        None

    def ica(self,imagedata):
        self.times.append(self.timer.timeSinceStart())
        c_means = self.getMean(imagedata)
        self.frame_buffers.append(c_means)
        
        buffer_len = len(self.frame_buffers)
        if buffer_len > self.max_buffer_size:
            self.frame_buffers = self.frame_buffers[-self.max_buffer_size:]
            self.times = self.times[-self.max_buffer_size:]
            buffer_len = self.max_buffer_size
        
        frame_data = np.array(self.frame_buffers)
        
        if buffer_len> 10:
            self.output_dim = frame_data.shape[0]
            fps = float(buffer_len) / (self.times[len(self.times)-1] - self.times[0])
            time_steps = np.linspace(self.times[0], self.times[len(self.times)-1], buffer_len)
            ####       
            interpolated = np.interp(time_steps, self.times, frame_data)
            interpolated = np.hamming(buffer_len) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(fps) / buffer_len * np.arange(buffer_len / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > MIN_BMP) & (freqs < MAX_BMP))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned

            idx2 = np.argmax(pruned)
            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1
        
        
        
        
        
        print("ica")
        print(self.bpm)
        

    def getMean(self, imagedata):
        c1 = np.mean(imagedata[:,:,0])
        c2 = np.mean(imagedata[:,:,1])
        c3 = np.mean(imagedata[:,:,2])
        return (c1+c2+c3)/3.0