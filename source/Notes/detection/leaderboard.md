
- YOLOv3

| Model                                                        | APval | APtest | AP50 | SpeedGPU | FPSGPU |      | params | FLOPS  |
| ------------------------------------------------------------ | ----- | ------ | ---- | -------- | ------ | ---- | ------ | ------ |
| [YOLOv3](https://github.com/ultralytics/yolov3/releases)     | 43.3  | 43.3   | 63   | 4.8ms    | 208    |      | 61.9M  | 156.4B |
| [YOLOv3-SPP](https://github.com/ultralytics/yolov3/releases) | 44.3  | 44.3   | 64.6 | 4.9ms    | 204    |      | 63.0M  | 157.0B |
| [YOLOv3-tiny](https://github.com/ultralytics/yolov3/releases) | 17.6  | 34.9   | 34.9 | 1.7ms    | 588    |      | 8.9M   | 13.3B  |

- YOLOv5

| Model                                                        | size (pixels) | mAPval 50-95 | mAPval 50 | Speed CPU b1 (ms) | Speed V100 b1 (ms) | Speed V100 b32 (ms) | params (M) | FLOPs @640 (B) |
| ------------------------------------------------------------ | ------------- | ------------ | --------- | ----------------- | ------------------ | ------------------- | ---------- | -------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt) | 640           | 28           | 45.7      | 45                | 6.3                | 0.6                 | 1.9        | 4.5            |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) | 640           | 37.4         | 56.8      | 98                | 6.4                | 0.9                 | 7.2        | 16.5           |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt) | 640           | 45.4         | 64.1      | 224               | 8.2                | 1.7                 | 21.2       | 49             |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt) | 640           | 49           | 67.3      | 430               | 10.1               | 2.7                 | 46.5       | 109.1          |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt) | 640           | 50.7         | 68.9      | 766               | 12.1               | 4.8                 | 86.7       | 205.7          |
|                                                              |               |              |           |                   |                    |                     |            |                |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt) | 1280          | 36           | 54.4      | 153               | 8.1                | 2.1                 | 3.2        | 4.6            |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt) | 1280          | 44.8         | 63.7      | 385               | 8.2                | 3.6                 | 12.6       | 16.8           |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt) | 1280          | 51.3         | 69.3      | 887               | 11.1               | 6.8                 | 35.7       | 50             |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt) | 1280          | 53.7         | 71.3      | 1784              | 15.8               | 10.5                | 76.8       | 111.4          |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt) + [TTA](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation) | 1280 1536     | 55.0 55.8    | 72.7 72.7 | 3136 -            | 26.2 -             | 19.4 -              | 140.7 -    | 209.8 -        |

- YOLOv8

| Model                                                        | size (pixels) | mAPval 50-95 |      |      | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | params (M) | FLOPs (B) |
| ------------------------------------------------------------ | ------------- | ------------ | ---- | ---- | ------------------- | ------------------------ | ---------- | --------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 640           | 37.3         |      |      | 80.4                | 0.99                     | 3.2        | 8.7       |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | 640           | 44.9         |      |      | 128.4               | 1.2                      | 11.2       | 28.6      |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 640           | 50.2         |      |      | 234.7               | 1.83                     | 25.9       | 78.9      |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | 640           | 52.9         |      |      | 375.2               | 2.39                     | 43.7       | 165.2     |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | 640           | 53.9         |      |      | 479.1               | 3.53                     | 68.2       | 257.8     |

- YOLOv6

| Model                                                        | Size | mAPval 0.5:0.95 |      |      | SpeedT4 trt fp16 b1 (fps) | SpeedT4 trt fp16 b32 (fps) | Params (M) | FLOPs (G) |
| ------------------------------------------------------------ | ---- | --------------- | ---- | ---- | ------------------------- | -------------------------- | ---------- | --------- |
| [YOLOv6-N](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt) | 640  | 37.5            |      |      | 779                       | 1187                       | 4.7        | 11.4      |
| [YOLOv6-S](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt) | 640  | 45              |      |      | 339                       | 484                        | 18.5       | 45.3      |
| [YOLOv6-M](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt) | 640  | 50              |      |      | 175                       | 226                        | 34.9       | 85.8      |
| [YOLOv6-L](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt) | 640  | 52.8            |      |      | 98                        | 116                        | 59.6       | 150.7     |
|                                                              |      |                 |      |      |                           |                            |            |           |
| [YOLOv6-N6](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n6.pt) | 1280 | 44.9            |      |      | 228                       | 281                        | 10.4       | 49.8      |
| [YOLOv6-S6](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s6.pt) | 1280 | 50.3            |      |      | 98                        | 108                        | 41.4       | 198       |
| [YOLOv6-M6](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m6.pt) | 1280 | 55.2            |      |      | 47                        | 55                         | 79.6       | 379.5     |
| [YOLOv6-L6](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l6.pt) | 1280 | 57.2            |      |      | 26                        | 29                         | 140.4      | 673.4     |

- YOLOv7

| Model                                                        | Test Size | APtest | AP50test | AP75test | batch 1 fps | batch 32  average time |
| ------------------------------------------------------------ | --------- | ------ | -------- | -------- | ----------- | ---------------------- |
| [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640       | 51.40% | 69.70%   | 55.90%   | 161 fps     | 2.8 ms                 |
| [YOLOv7-X](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640       | 53.10% | 71.20%   | 57.80%   | 114 fps     | 4.3 ms                 |
|                                                              |           |        |          |          |             |                        |
| [YOLOv7-W6](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280      | 54.90% | 72.60%   | 60.10%   | 84 fps      | 7.6 ms                 |
| [YOLOv7-E6](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280      | 56.00% | 73.50%   | 61.20%   | 56 fps      | 12.3 ms                |
| [YOLOv7-D6](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280      | 56.60% | 74.00%   | 61.80%   | 44 fps      | 15.0 ms                |
| [YOLOv7-E6E](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280      | 56.80% | 74.40%   | 62.10%   | 36 fps      | 18.7 ms                |

- YOLOv9

| Model                                                        | Test Size | APval  | AP50val | AP75val | Param. | FLOPs  |
| ------------------------------------------------------------ | --------- | ------ | ------- | ------- | ------ | ------ |
| [YOLOv9-T](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) | 640       | 38.30% | 53.10%  | 41.30%  | 2.0M   | 7.7G   |
| [YOLOv9-S](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt) | 640       | 46.80% | 63.40%  | 50.70%  | 7.1M   | 26.4G  |
| [YOLOv9-M](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt) | 640       | 51.40% | 68.10%  | 56.10%  | 20.0M  | 76.3G  |
| [YOLOv9-C](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt) | 640       | 53.00% | 70.20%  | 57.80%  | 25.3M  | 102.1G |
| [YOLOv9-E](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt) | 640       | 55.60% | 72.80%  | 60.60%  | 57.3M  | 189.0G |

- YOLOv10

| Model                                                  | Test Size | APval  |      |      | #Params | FLOPs  | Latency |
| ------------------------------------------------------ | --------- | ------ | ---- | ---- | ------- | ------ | ------- |
| [YOLOv10-N](https://huggingface.co/jameslahm/yolov10n) | 640       | 38.50% |      |      | 2.3M    | 6.7G   | 1.84ms  |
| [YOLOv10-S](https://huggingface.co/jameslahm/yolov10s) | 640       | 46.30% |      |      | 7.2M    | 21.6G  | 2.49ms  |
| [YOLOv10-M](https://huggingface.co/jameslahm/yolov10m) | 640       | 51.10% |      |      | 15.4M   | 59.1G  | 4.74ms  |
| [YOLOv10-B](https://huggingface.co/jameslahm/yolov10b) | 640       | 52.50% |      |      | 19.1M   | 92.0G  | 5.74ms  |
| [YOLOv10-L](https://huggingface.co/jameslahm/yolov10l) | 640       | 53.20% |      |      | 24.4M   | 120.3G | 7.28ms  |
| [YOLOv10-X](https://huggingface.co/jameslahm/yolov10x) | 640       | 54.40% |      |      | 29.5M   | 160.4G | 10.70ms |

- PP-YOLO

| Model      | GPU number | images/GPU | backbone    | input shape | Box  APval | Box  APtest | V100  FP32(FPS) | V100  TensorRT FP16(FPS) |
| ---------- | ---------- | ---------- | ----------- | ----------- | ---------- | ----------- | --------------- | ------------------------ |
| PP-YOLO    | 8          | 24         | ResNet50vd  | 608         | 44.8       | 45.2        | 72.9            | 155.6                    |
| PP-YOLO    | 8          | 24         | ResNet50vd  | 512         | 43.9       | 44.4        | 89.9            | 188.4                    |
| PP-YOLO    | 8          | 24         | ResNet50vd  | 416         | 42.1       | 42.5        | 109.1           | 215.4                    |
| PP-YOLO    | 8          | 24         | ResNet50vd  | 320         | 38.9       | 39.3        | 132.2           | 242.2                    |
| PP-YOLO_2x | 8          | 24         | ResNet50vd  | 608         | 45.3       | 45.9        | 72.9            | 155.6                    |
| PP-YOLO_2x | 8          | 24         | ResNet50vd  | 512         | 44.4       | 45          | 89.9            | 188.4                    |
| PP-YOLO_2x | 8          | 24         | ResNet50vd  | 416         | 42.7       | 43.2        | 109.1           | 215.4                    |
| PP-YOLO_2x | 8          | 24         | ResNet50vd  | 320         | 39.5       | 40.1        | 132.2           | 242.2                    |
| PP-YOLO    | 4          | 32         | ResNet18vd  | 512         | 29.2       | 29.5        | 357.1           | 657.9                    |
| PP-YOLO    | 4          | 32         | ResNet18vd  | 416         | 28.6       | 28.9        | 409.8           | 719.4                    |
| PP-YOLO    | 4          | 32         | ResNet18vd  | 320         | 26.2       | 26.4        | 480.7           | 763.4                    |
| PP-YOLOv2  | 8          | 12         | ResNet50vd  | 640         | 49.1       | 49.5        | 68.9            | 106.5                    |
| PP-YOLOv2  | 8          | 12         | ResNet101vd | 640         | 49.7       | 50.3        | 49.5            | 87                       |

- PP-YOLOE

| Model       | Epoch | GPU number | images/GPU | backbone    | input shape | Box APval 0.5:0.95 | Box APtest 0.5:0.95 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |
| ----------- | ----- | ---------- | ---------- | ----------- | ----------- | ------------------ | ------------------- | --------- | -------- | -------------- | ----------------------- |
| PP-YOLOE+_s | 80    | 8          | 8          | cspresnet-s | 640         | 43.7               | 43.9                | 7.93      | 17.36    | 208.3          | 333.3                   |
| PP-YOLOE+_m | 80    | 8          | 8          | cspresnet-m | 640         | 49.8               | 50                  | 23.43     | 49.91    | 123.4          | 208.3                   |
| PP-YOLOE+_l | 80    | 8          | 8          | cspresnet-l | 640         | 52.9               | 53.3                | 52.2      | 110.07   | 78.1           | 149.2                   |
| PP-YOLOE+_x | 80    | 8          | 8          | cspresnet-x | 640         | 54.7               | 54.9                | 98.42     | 206.59   | 45             | 95.2                    |









