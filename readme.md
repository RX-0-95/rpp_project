# PyQT based RPPG project 

## About the project 
A PyQT GUI project use ICA algorithms to remotely detect human heart rate. The average error in effective detection period is +-2 bpms. The program is able to auto detect face and auto redetect face when face moved during detection. The prgram can either perform real time detection though webcam or detect heart rate though a video. 

## How to run the prgram

1. Set python virtual evn (recommand not necessary)


2. install all packages in installed_packages.txt or simply use command 

            pip install -r requirements.txt 

3. In terminal 

            python3 run.py 



## To Do 
- [x] forehead detection 
- [x] whole face crop 
- [x] App open camera button to main interface 
- [x] add open video file button to main interface 
- [x] add change camera function 
- [x] abstract alogrithm class 
- [x] ica alogorithm
- [x] face move detection 
- [x] re-detect face and forehead after face move  
- [ ] ssr alogorthm 
- [x] video mode 