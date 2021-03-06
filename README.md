# Escooter_No_Helmet_Detection_YOLO

**weight file** + **project video** can be found  [HERE](https://drive.google.com/drive/folders/1d93KtB0RRVNFkZVwE1w5qEKabRfuCBcY?usp=sharing)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NVuJhiyGVc8/0.jpg)](https://youtu.be/NVuJhiyGVc8)


## YOLO
model trained on self collected 5000 e_scooter custom dataset
- model trained with [Darknet framework](https://github.com/pjreddie/darknet)
- dataset labeled using [yolo_mark](https://github.com/AlexeyAB/Yolo_mark)


## Django
put weight file under hackathon/

a webpage that can display and track helmet/no_helmet counts on different streets

![image](https://user-images.githubusercontent.com/67103130/141060294-7ade4dfd-f965-47a0-a521-b055a6fe6ebd.png)


## Object Counting
use [SORT](https://arxiv.org/pdf/1703.07402.pdf) algorithm

code template from [HERE](https://github.com/HodenX/python-traffic-counter-with-yolo-and-sort)

```
python main.py --input test_video.mp4 --output output.mp4
```


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/bbWIGhTZly0/0.jpg)](https://youtu.be/bbWIGhTZly0)

