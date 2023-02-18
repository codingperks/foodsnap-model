from ultralytics import YOLO
import numpy as np

def detect_food_items(input):
    """
    Generate food class prediction and item coordinates
    using YOLO model.
    Args:
        input (Path): Path to image for classification.
    Returns:
        dims (NDArray): (N, 2) array containing normalised
            height and width values.
        labels (NDArray): (1, N) array containing food items.
    """
    # load pre-trained model
    model = YOLO("./02_local_testing/seg_50_omelette.pt")

    # generate prediction based on input image
    # classes: classes included in search (0 = bowl, 1 = omelette, 2 = plate)
    results = model.predict(source=input)

    # get identified classes
    cls_tensor = results[0].boxes.cls

    # create numpy label array
    cls_np = np.expand_dims(cls_tensor.detach().numpy(), axis=0)
    names_dict = model.names
    to_np = np.vectorize(lambda i: names_dict[i])
    labels = (np.asarray(to_np(cls_np))).flatten()

    # extract mask element
    mask_data = results[0].masks.data # raw masks tensor (N, H, W)

    # create empty np array for object areas
    area_np = np.empty(shape=labels.size, dtype=float)

    # iterate through elements in tensor and extract mask size
    for i in range(mask_data.shape[0]):
        mask_zero = (mask_data[i] == 0).sum()
        mask_one = (mask_data[i] == 1).sum()
        area_tensor = mask_one / (mask_zero + mask_one)
        area_np[i] = area_tensor.detach().numpy()

    return area_np, labels

print(detect_food_items("./02_local_testing/omelette_1.jpeg"))
