from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="//Users/aadi/PycharmProjects/TrainModel/DataForArmyV2_removing_Imbalance/data.yaml",
    imgsz=832,
    batch=16,
    epochs=80,
    patience=5,
    device="mps",
    cls=1.8,
    multi_scale=True,
    rect=True,
    mosaic=0.3,
    close_mosaic=10,
    mixup=0.05,
    workers=4,
    cache=True,
    amp=True,
    name="yolov8s_balanced"
)

# model = YOLO("yolov8n.pt")
# results = model.train(
#     data="/Users/aadi/PycharmProjects/TrainModel/DataForArmyV1/data.yaml",
#     imgsz=512,
#     epochs=20,
#     batch=32,
#     device="mps",
#     workers=4,
#     cache=True,
#     amp=True,
#     mosaic=0.0,
#     mixup=0.0,
#     name="yolov8n_fast_and_custom"
# )


# need to find more optimal values for this