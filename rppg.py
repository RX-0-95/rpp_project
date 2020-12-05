from os import terminal_size
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtMultimedia as qtm
from PyQt5.QtCore import pyqtSlot
from numpy import ndarray
import numpy as np
from timer import *
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Class rppg is a singleton
# part of the code from ##TODO: citation

MIN_BMP = 40
MAX_BMP = 200
MAX_BUFFER_SIZE = 900


class rppg:
    __instance = None

    @staticmethod
    def rppgInstance():
        if rppg.__instance == None:
            rppg()
        return rppg.__instance

    def __init__(self):
        if rppg.__instance != None:
            raise Exception("rppg Class is a singleton class")
        else:
            rppg.__instance = self
            self.image = None
            self.timer = timer.timerInstance()
            self.frame_buffers = []
            self.max_buffer_size = MAX_BUFFER_SIZE
            self.times = []
            self.output_dim = 0
            self.freqs = []
            self.fft = []
            self.bpms = []
            self.bpm = 0
            self.idx = 1
            self.frame_step = 0
            self.videoMode = False
            self.cameraMode = True
            self.bmpdatas = []
            self.plt_idx = 0

    def setVideoMode(self, frame_step):
        self.cameraMode = False
        self.videoMode = True
        self.frame_step = frame_step

    def setCameraMode(self):
        self.cameraMode = True
        self.videoMode = False
        self.frame_step = 0

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

    def ica(self, imagedata):
        if self.cameraMode:
            self.times.append(self.timer.timeSinceStart())
        elif self.videoMode:
            if len(self.times) == 0:
                # self.times.append(self.frame_step)
                self.times.append(self.frame_step)
            else:
                self.times.append((self.times[-1] + self.frame_step))
                # print("time step " + (str(self.frame_step)) )
                # print(self.times)
        # c_means = self.getMean(imagedata)
        # print(self.times)
        c_means = imagedata
        print(c_means)
        self.frame_buffers.append(c_means)

        buffer_len = len(self.frame_buffers)
        if buffer_len > self.max_buffer_size:
            self.frame_buffers = self.frame_buffers[-self.max_buffer_size :]
            # print(self.frame_buffers)
            self.times = self.times[-self.max_buffer_size :]
            buffer_len = self.max_buffer_size

        frame_data = np.array(self.frame_buffers)

        if buffer_len > 10:
            self.output_dim = frame_data.shape[0]
            # print(self.times)
            fps = float(buffer_len) / (self.times[-1] - self.times[0])
            print("fps" + str(fps))
            time_steps = np.linspace(self.times[0], self.times[-1], buffer_len)
            # print("time step")
            # print(time_steps)
            ####
            interpolated = np.interp(time_steps, self.times, frame_data)
            interpolated = np.hamming(buffer_len) * interpolated
            interpolated = interpolated - np.mean(interpolated)

            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)

            if self.cameraMode:
                self.freqs = float(fps) / buffer_len * np.arange(buffer_len / 2 + 1)
            elif self.videoMode:
                # print("Video Mode-********************")
                self.freqs = np.fft.fftfreq(len(interpolated), 1.0 / fps)
            # print("self freq")
            # print(self.freqs)
            freqs = 60.0 * self.freqs
            # print("++++++++++++freqs++++++++++++")
            # print(freqs)
            idx = np.where((freqs > MIN_BMP) & (freqs < MAX_BMP))

            pruned = self.fft[idx]

            phase = phase[idx]
            ##for plot
            self.plt_idx = idx

            ##end draw fft
            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            """
            print("++++++++++++idx+++++++++++")
            print(self.idx)
            print("-----------pruned----------")
            print(pruned)
            print("++++++++++++freqs++++++++++++")
            print(self.freqs)
            """
            if len(pruned) >= 10:
                pruned[0] = 0
                pruned[1] = 0
                pruned[2] = 0
                pruned[3] = 0
                pruned[4] = 0
                pruned[5] = 0
                pruned[6] = 0
                pruned[7] = 0
                pruned[8] = 0
                pruned[9] = 0

            idx2 = np.argmax(pruned)
            t = (np.sin(phase[idx2]) + 1.0) / 2.0
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            self.bpm = self.freqs[idx2]
            self.idx += 1
            gap = (self.max_buffer_size - buffer_len) / fps

            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
                print(text)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
                print(text)

            self.bmpdatas.append(self.bpm)

        print("ica")

    def getMean(self, imagedata):
        # print("Image Data size")
        # print(imagedata.shape)
        c1 = 1.2 * np.mean(imagedata[:, :, 0])
        c2 = np.mean(imagedata[:, :, 1])
        c3 = np.mean(imagedata[:, :, 2])
        return (c1 + c2 + c3) / 3.0
