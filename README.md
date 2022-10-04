# UAVDetection
T-Brain_無人機飛行載具之智慧計數競賽

* 10/3 官方訓練及測試資料發布 分數為0.548014
* 10/4 官方訓練及測試資料發布 分數為


---
* 訓練
```
python train.py --data data/Drone.yaml --cfg models/yolov5x_Drone.yaml --hyp data/hyps/hyp.scratch-high_Drone.yaml --img-size 640 --batch 8 --epochs 300 --weights yolov5x.pt --image-weight --cache --label-smoothing 0.01
```

* 輸出偵測結果:
