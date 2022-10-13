# UAVDetection
T-Brain_無人機飛行載具之智慧計數競賽

* 10/3 官方訓練及測試資料發布 分數為0.548014 (1003-1)
* 10/3 加入VisDrone 2022 分數為 0.582223 (1003-2)
* 10/5 加入Albumentation演算法進行資料擴增、scale改0.3(原0.9) 分數為 0.557835
* 10/6 對輸入影像進行平舖、scale改回0.9 分數為 0.584564
* 10/11 對輸入影像進行平舖、調整遇預測信心分數到.5 分數為 0.653412
* 10/11 對輸入影像進行平舖、調整遇預測信心分數到.3 分數為 0.580613
* slice_height 384 、 overlap_height_ratio 0.2、threshold 0.5: 0.660452
```
資料擴增參數
A.RandomGamma(p=0.1),
A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.1), #將對比度受限的自適應直方圖均衡應用於輸入圖像。
A.RandomBrightnessContrast(p=0.1),
A.RandomToneCurve(p=0.1),
A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.1), #通過縮小和放大回來降低圖像質量
A.FancyPCA(alpha=0.1, always_apply=False, p=0.1), #增強 RGB 圖像
A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.1), #隨機改變圖像的亮度、對比度和飽和度
A.Sharpen(p=0.1),
```
* 10/? 重新label官方訓練集 分數為 X

---


- [ ] YOLOv5 + SAHI

---

* SAHI環境安裝
```
pip install sahi
pip install yolov5
pip install pandas
```

---
* 訓練
```
python train.py --data data/Drone.yaml --cfg models/yolov5x_Drone.yaml --hyp data/hyps/hyp.scratch-high_Drone.yaml --img-size 640 --batch 8 --epochs 300 --weights yolov5x.pt --image-weight --cache --label-smoothing 0.01
```

* 輸出偵測結果+.csv:
> 
[https://github.com/gcobs051094/UAVDetection/blob/main/SAHI_Detection_for_YOLOv5.ipynb](https://)
