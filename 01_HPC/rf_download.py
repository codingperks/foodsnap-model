from roboflow import Roboflow
rf = Roboflow(api_key="GjP8E52qNp1AUWdM04dK")
project = rf.workspace("").project("foodsnap_data")
dataset = project.version(1).download("yolov8")
