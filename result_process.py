import csv
from sys import prefix
from rppg import rppg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde

def processGroundData(file_name,second_unit = 10000):
    ground_relative_time = []
    ground_heart_rate = []
    with open(file_name) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            _, relative_time, heart_rate = row
            ground_relative_time.append(relative_time)
            ground_heart_rate.append(heart_rate)

    ground_relative_time = ground_relative_time[1:]
    ground_heart_rate = ground_heart_rate[1:]
    ground_relative_time = list(map(int, ground_relative_time))
    ground_heart_rate = list(map(float, ground_heart_rate))
    ground_time = [(x - ground_relative_time[0]) /second_unit for x in ground_relative_time]
    ground_lenght = int(ground_time[-1])
    ground_time = np.array(ground_time)
    ground_heart_rate = np.array(ground_heart_rate)
    time = []
    heart_rate = []
    for i in range(0, ground_lenght):
        idx = np.where((ground_time >= i) & (ground_time <= i + 1))
        bmp = np.average(ground_heart_rate[idx])
        time.append(i + 1)
        heart_rate.append(bmp)

    return time, heart_rate

def processRppgData(file_name,fps):
    frame_list = []
    heart_rate_list = []
    with open(file_name, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            frame, heart_rate = row
            frame_list.append(frame)
            heart_rate_list.append(heart_rate)

    frame_list = list(map(int, frame_list))
    heart_rate_list = list(map(float, heart_rate_list))
    heart_rate_rec = []
    seconds = []
    heart_rate_persecond = []
    for i, frame in enumerate(frame_list):
        heart_rate_rec.append(heart_rate_list[i])
        if frame % fps == 0:
            if len(heart_rate_rec) == 0:
                heart_rate_persecond.append(0)
            else:
                heart_rate_persecond.append(sum(heart_rate_rec) / len(heart_rate_rec))
            seconds.append(int(frame / fps))
            heart_rate_rec = []
    return seconds, heart_rate_persecond

def dataDiff(rppgData,groundData):
    ret = []
    _rppgData = np.array(rppgData)
    _groundData = np.array(groundData)
    for i, rate in enumerate(_rppgData):
        ret.append(_groundData[i]-rate)
    return ret 

def percentWithinRange(rppgdata,groundData,range):
    diff = dataDiff(rppgdata, groundData)
    diff = np.array(diff)
    diff = diff[np.logical_not(np.isnan(diff))]
    idx = np.where((diff>=range[0])&(diff<=range[1]))
    if len(rppgdata)==0:
        return False 
    else:
        return 100*len(idx[0])/len(diff)


ground_filename = "MPDataExport.csv"
output_filename = "video_front_1_output.csv"

fig, axs = plt.subplots(2,2)
seconds, heart_rate_persecond = processRppgData(output_filename, 30)
ground_time, ground_heart_rate = processGroundData(ground_filename)
print(ground_heart_rate)

##Process for my data 
"""
window_size  = 50
ground_time = ground_time[window_size:]
ground_heart_rate = ground_heart_rate[window_size:]
"""

ground_length = len(ground_time)
rppg_time = seconds[-ground_length:]
rppg_heart_rate = heart_rate_persecond[-ground_length:]
ground_time = [(x + rppg_time[0]) for x in ground_time]

print("lenght of ground data" + str(len(ground_time)))
print("length of ground time"+ str(len(ground_heart_rate)))
print("length of estimate data" + str(len(rppg_time)))
diff = [] 
for i, rate in enumerate(rppg_heart_rate):
    diff.append(ground_heart_rate[i]-rate)


# axs[1].plot(time, heart_rate)
diff = np.array(diff)
diff_without_nan = diff[np.logical_not(np.isnan(diff))]
print(diff_without_nan)
print(np.std(diff_without_nan))
print(np.mean(diff_without_nan))
print(percentWithinRange(rppg_heart_rate,ground_heart_rate, [-5,5]))
#plot pdf 

kde = gaussian_kde(diff_without_nan)
dist_space = np.linspace( min(diff_without_nan), max(diff_without_nan), 100 )
axs[0,0].plot(rppg_time, rppg_heart_rate)
axs[0,0].set_title("Ground Truth", fontsize = 15)
axs[0,0].set_xlabel("Time in video(seconds)", fontsize = 15)
axs[0,0].set_ylabel("Heart rate", fontsize = 15)
#, ylabel = "Heart rate")
axs[0,1].plot(ground_time, ground_heart_rate)
axs[0,1].set_title("Estimate Heart Rate", fontsize = 15)
axs[0,1].set_xlabel("Time in video(seconds)",fontsize = 15) 
axs[0,1].set_ylabel("Heart rate",fontsize = 15) 
axs[1,0].plot(ground_time,diff)
axs[1,0].set_title("Difference\Error", fontsize = 15)
axs[1,0].set_xlabel("Time in video(seconds)",fontsize = 15) 
axs[1,0].set_ylabel("Difference/Error in heart rate",fontsize = 15) 
axs[1,1].plot(dist_space, kde(dist_space))
axs[1,1].set_title("PDF of Differnce\Error", fontsize = 15)
axs[1,1].set_xlabel("Difference\Error",fontsize = 15) 
axs[1,1].set_ylabel("Probability",fontsize = 15) 


plt.show()

