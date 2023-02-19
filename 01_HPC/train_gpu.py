from ultralytics import YOLO
import time

#loading pre-trained model
#parameter: yolo8[n/s/m/l/x] for size
model = YOLO("yolov8s-seg.pt")

#report training start
print("Training starting")

#start time measurement
time_start = time.perf_counter()

#training model
#param data = training data - standard path: ""./training_data/data.yaml"
#param epochs = epochs (whole runs over dataset)
#param imgsz = image size (pixels)
#param patience = epochs to wait for no observable improvement (to stop early)
#param cache = caching (default false) with possible values (True, False, ram, disk)
#param device = device to run on (cuda 0,1,N) or cpu - take out if cpu training, device=0,1,2,3,4,5,6,7
results = model.train(data='/rds/general/user/jcp22/home/foodsnap_data/data.yaml', epochs=100, imgsz=640, patience=25, device=0)

#arguments for experimentation
#device=0
#cache=True
#optimizer='Adam'
#workers=16
#batch=-1
#data='$PBS_O_WORKDIR/fs_data_01/data.yaml'

#stop time measurement and report result
time_elapsed = (time.perf_counter() - time_start)/60
print("The task took ", round(time_elapsed, 2), " minutes")
