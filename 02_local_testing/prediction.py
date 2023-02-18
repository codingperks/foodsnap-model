from ultralytics import YOLO

#loading pre-trained model
#model = YOLO("path/to/model.pt")
model = YOLO("./02_local_testing/seg_50_omelette.pt")

#make prediction
#param source = image to predict contents of
#save = save model output as .png
results = model.predict(source="./02_local_testing/omelette_1.jpeg", save=True)
