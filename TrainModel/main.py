from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="/workspace/DataForArmyV2_removing_Imbalance/data.yaml",  # contains class_weights

    imgsz=832,
    batch=48,
    epochs=120,
    patience=15,

    device=0,            # CUDA (RTX 4090)
    amp=True,

    optimizer="AdamW",
    lr0=0.003,
    lrf=0.1,
    weight_decay=0.0005,

    # Augmentations (balanced for recall + precision)
    mosaic=0.15,
    close_mosaic=20,
    mixup=0.1,
    multi_scale=True,

    # IMPORTANT: remove global cls boost
    cls=1.0,

    # Performance
    workers=12,
    cache="disk",

    val=True,
    plots=True,

    name="yolov8m_runpod_weighted"
)


# from ultralytics import YOLO
#
# model = YOLO("yolov8s.pt")
# model.train(
#     data="//Users/aadi/PycharmProjects/TrainModel/DataForArmyV2_removing_Imbalance/data.yaml",
#     imgsz=832,
#     batch=16,
#     epochs=80,
#     patience=5,
#     device="mps",
#     cls=1.8,
#     multi_scale=True,
#     rect=True,
#     mosaic=0.3,
#     close_mosaic=10,
#     mixup=0.05,
#     workers=4,
#     cache=True,
#     amp=True,
#     name="yolov8s_balanced"
# )

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