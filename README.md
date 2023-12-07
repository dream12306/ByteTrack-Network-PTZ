# README——基于yoloX和ByteTrack的人体识别与追踪

## 一、小组信息

### 成员信息

####  刘向前U202112570（组长），陆奕丞 U202111463，陈庭柱 U202111740。

### 组内分工

1. 刘向前：主程序的设计，图像识别部分，GUI界面部分。
2. 陆奕丞：摄像头控制部分。
3. 陈庭柱：网络云台图像获取部分。

## 二、项目介绍

​		本项目，旨在利用BYTETRACK框架与yolo模型，找出摄像头读取到的视频帧中所有的行人，包括位置和大小，并用矩形框与唯一id标识出来。在此基础上，可以人为选中跟踪目标做出特别标识，目前设计的选定方式为鼠标左键点击相应的检测框，取消选定的方式为鼠标右键点击相应的检测框或特定按钮。除此之外，界面还提供了控制摄像头移动，放大倍数，自动变焦等按钮。具体设定如下：

![按钮及功能](D:\1_dream_person\小学期\招商证券人工智能软件工程训练营\第二阶段\多进程\ByteTrack\按钮及功能.jpg)

### 项目文件夹

项目框架克隆于[[ifzhang/ByteTrack: [ECCV 2022\] ByteTrack: Multi-Object Tracking by Associating Every Detection Box (github.com)](https://github.com/ifzhang/ByteTrack)](https://github.com/NirAharon/BoT-SORT)

#### 1.小组作业所在文件夹：

我们小组基于BYTETRACK项目框架编写的程序入口为

`"ByteTrack\tools\mian.py"`

图像识别部分python文件为

`"ByteTrack\tools\demo_track.py"`

摄像头控制部分python文件为

`"ByteTrack\tools\camera.py"`

网络云台图像获取部分python文件为

`"ByteTrack\tools\ffmpeg_my.py"`

#### 2.YOLO模型文件夹：

本程序使用的模型为yolox，所用到的与模型相关的工具函数或者类存在于

`“ByteTrack/yolox“`

#### 3.程序使用的模型文件：

`ByteTrack/pretrained/yolox_s.pth`

`ByteTrack/pretrained/bytetrack_s_mot17.pth.tar`

#### 4.课程报告位置：

##### 见“ByteTrack/”根目录“基于yolox和ByteTrack的人体识别与跟踪—课程报告”

## 三、基础环境

### Windows 11

### python=3.9.18

### 使用的库：

```
absl-py                 1.4.0
aiohttp                 3.8.5
aiosignal               1.2.0
archspec                0.2.1
async-timeout           4.0.2
attrs                   23.1.0
blinker                 1.6.2
boltons                 23.0.0
Brotli                  1.0.9
cachetools              4.2.2
certifi                 2023.11.17
cffi                    1.16.0
charset-normalizer      2.0.4
click                   8.1.7
cloudpickle             2.2.1
colorama                0.4.6
coloredlogs             15.0.1
conda                   23.10.0
conda-libmamba-solver   23.11.1
conda-package-handling  2.2.0
conda_package_streaming 0.9.0
contourpy               1.2.0
cPython                 0.0.6
cryptography            41.0.3
cycler                  0.12.1
Cython                  3.0.5
cython-bbox             0.1.5
cytoolz                 0.12.0
dask                    2023.6.0
dill                    0.3.7
dnspython               2.4.2
ffmpeg-python           0.2.0
filelock                3.13.1
filterpy                1.4.5
flatbuffers             23.5.26
fonttools               4.45.0
frozenlist              1.4.0
fsspec                  2023.9.2
future                  0.18.3
google-auth             2.22.0
google-auth-oauthlib    0.4.4
grpcio                  1.48.2
h5py                    3.10.0
humanfriendly           10.0
idna                    3.4
imagecodecs             2023.1.23
imageio                 2.31.4
importlib-metadata      6.0.0
importlib-resources     6.1.1
Jinja2                  3.1.2
jsonpatch               1.32
jsonpointer             2.1
kiwisolver              1.4.5
lap                     0.4.0
libmambapy              1.5.3
locket                  1.0.0
loguru                  0.5.3
Markdown                3.4.1
markdown-it-py          3.0.0
MarkupSafe              2.1.1
matplotlib              3.8.2
mdurl                   0.1.2
menuinst                1.4.19
mkl-fft                 1.3.8
mkl-random              1.2.4
mkl-service             2.4.0
motmetrics              1.4.0
mpmath                  1.3.0
multidict               6.0.2
multiprocess            0.70.15
networkx                3.1
numpy                   1.23.5
oauthlib                3.2.2
onnx                    1.15.0
onnx-simplifier         0.4.35
onnxruntime             1.16.3
opencv-python           4.8.1.78
packaging               23.1
pandas                  2.1.3
partd                   1.4.1
Pillow                  10.0.1
pip                     23.3
pluggy                  1.0.0
protobuf                3.20.3
pyasn1                  0.4.8
pyasn1-modules          0.2.8
pycocotools             2.0.7
pycosat                 0.6.6
pycparser               2.21
Pygments                2.17.2
PyJWT                   2.4.0
pymongo                 4.6.0
pyOpenSSL               23.2.0
pyparsing               3.1.1
pyreadline3             3.4.1
pyserial                3.5
PySocks                 1.7.1
python-dateutil         2.8.2
pytz                    2023.3.post1
PyWavelets              1.4.1
PyYAML                  6.0.1
requests                2.31.0
requests-oauthlib       1.3.0
rich                    13.7.0
rsa                     4.7.2
ruamel.yaml             0.17.21
ruamel.yaml.clib        0.2.6
scikit-image            0.19.3
scipy                   1.11.3
sympy                   1.11.1
tensorboard             2.10.0
tensorboard-plugin-wit  1.8.1
thop                    0.1.1.post2209072238
tifffile                2023.4.12
toolz                   0.12.0
torch                   2.1.0
torchaudio              2.1.0
torchvision             0.16.0
tqdm                    4.65.0
typing_extensions       4.7.1
tzdata                  2023.3
urllib3                 1.26.18
Werkzeug                2.2.3
wheel                   0.41.2
win-inet-pton           1.1.0
win32-setctime          1.1.0
xmltodict               0.13.0
yarl                    1.8.1
zipp                    3.11.0
zstandard               0.19.0
```

## 四、运行方式

### 摄像头实时识别与追踪：

在“BytrTrack/”文件夹内，在命令行窗口运行：

```
python tools/main.py webcam
```

## 五、程序相关

### 使用的重要函数：

#### main.py

```
# 非自定义函数
`BaseManager.register('VideoCapture', VideoCapture)`: 注册VideoCapture类到BaseManager，用于进程间通信。
`BaseManager.register('Magnify', Magnify)`: 注册Magnify类到BaseManager，用于进程间通信。
`Pool(processes=3)`: 创建一个进程池，最多同时执行3个进程。
`pool.apply_async(func=main, args=(exp, args, q_main_mag, size, q_main_rtsp))`: 异步执行main函数。
`pool.apply_async(func=run_mag, args=(mag,))`: 异步执行run_mag函数。
`pool.apply_async(func=run_rtsp, args=(rtsp,))`: 异步执行run_rtsp函数。

# 自定义函数
`run_mag(magnify)`: 用于运行视频放大处理的函数。
`run_rtsp(rtsp)`: 用于运行视频流处理的函数。
`main(exp, args, q_main_mag, size, q_main_rtsp)`: 初始化函数
```

#### camera.py

```
# 非自定义函数
`serial.Serial('com4', baudrate=19200, timeout=0.5)`: 用于获取串口对象
`ser.write()`: 用于向串口写入数据。
`ser.read()`: 用于从串口读取数据。

# 自定义函数
1. `setspeed(speed, ser, address, mode='x')`: 控制摄像头速度的函数，根据给定的速度和模式发送控制指令到摄像头。
2. `setspeedxy(speedx, speedy, ser, address)`: 控制摄像头在x和y方向的速度的函数，根据给定的速度发送控制指令到摄像头。
3. `setstop(ser, address)`: 控制摄像头停止运动的函数，发送停止指令到摄像头。
4. `moveto(position, ser, address, mode='x')`: 控制摄像头移动到指定位置的函数，根据给定的位置和模式发送控制指令到摄像头。
5. `queryPosition(ser, address, mode='x')`: 查询摄像头位置的函数，根据给定的模式发送查询指令到摄像头，并返回摄像头位置。
6. `setZoom(ser, address, zoom)`: 控制摄像头变焦的函数，根据给定的变焦值发送控制指令到摄像头。
7. `queryZoom(ser, address)`: 查询摄像头变焦值的函数，发送查询指令到摄像头，并返回变焦值。
```

#### ffmpeg_my.py

```
# 非自定义函数
`ffmpeg.probe()`: 用于获取音视频流的信息。
`ffmpeg.input()`: 用于指定输入文件或流。
`ffmpeg.output()`: 用于指定输出文件或流的格式和参数。
`ffmpeg.run_async()`: 用于异步执行ffmpeg命令。

# 自定义函数
`ffmpeg_init(source)`: 用于初始化ffmpeg处理音视频流的参数，并返回处理进程和视频流信息。
```

#### demo_track.py

```
1. `make_parser()`: 创建一个命令行参数解析器，用于解析运行脚本时传入的参数。
2. `get_image_list(path)`: 获取给定路径下的所有图像文件的列表。
3. `write_results(filename, results)`: 将跟踪结果写入到指定的文件中。
4. `class Predictor(object)`: 定义了一个预测器类，用于对输入的图像进行预测。
5. `Predictor.__init__(self, model, exp, trt_file=None, decoder=None, device=torch.device("cpu"), fp16=False)`: 预测器类的初始化函数，初始化模型、实验参数、设备等。
6. `Predictor.inference(self, img, timer)`: 预测器类的推理函数，对输入的图像进行预测，并返回预测结果和图像信息。
7. `image_demo(predictor, vis_folder, current_time, args)`: 对单个图像进行预测，并将预测结果保存和可视化。
8. `imageflow_demo(predictor, vis_folder, current_time, args,exp,queue_mag,size_screen,queue_rtsp)`: 对视频流进行预测，并将预测结果保存和可视化。
9. `update(vid_writer, current_time, args, results, frame_id, canvas, photo_image,exp,queue_mag,queue_rtsp)`: 在GUI界面上更新预测结果。
10. `on_click_left(event)`: 处理鼠标左键点击事件，将点击的目标添加到跟踪列表中。
11. `on_click_right(event)`: 处理鼠标右键点击事件，将点击的目标从跟踪列表中移除。
12. `startid()`: 开始目标检测。
13. `stopid()`: 停止目标检测。
14. `autozoom()`: 开始自动缩放。
15. `starttrack()`: 开始目标跟踪。
16. `stoptrack()`: 停止目标跟踪。
17. `cancelselect()`: 取消所有已选择的目标。
18. `up_press(event)`: 处理向上键按下事件，向上移动视角。
19. `down_press(event)`: 处理向下键按下事件，向下移动视角。
20. `left_press(event)`: 处理向左键按下事件，向左移动视角。
21. `right_press(event)`: 处理向右键按下事件，向右移动视角。
22. `release(event)`: 处理键盘释放事件，停止移动视角。
23. `cancel_tracking()`: 取消所有目标的跟踪。
24. `addzoom()`: 增加缩放倍数。
25. `subtractzoom()`: 减少缩放倍数。
```

### 重要自定义类

#### Magnify() # 实现对摄像头的控制

```
# 基本属性
		# 进程间通信队列
		self.queue = queue
        # 串口配置
        self.size = size
        self.ser = serial.Serial('com4', baudrate=19200, timeout=0.5)
        self.address = 1  # 设备地址
        # 摄像头初始化
        setZoom(self.ser,self.address,0)
        setstop(self.ser, self.address)
        # 画布中心XY坐标
        self.canvas_height = int(canvas_height)
        self.canvas_width = int(canvas_width)
        self.canvas_center_x = int(canvas_width / 2)
        self.canvas_center_y = int(canvas_height / 2)
		# 摄像头左右方向速度，画面x方向误差
        self.speedx = 0
        self.speedx_last = 0
        self.ex = 0
        self.ex_last1 = 0
        self.ex_last2 = 0
		# 摄像头上下方向速度，画面y方向误差
        self.speedy = 0
        self.speedy_last = 0
        self.ey = 0
        self.ey_last1 = 0
        self.ey_last2 = 0
        # 摄像头放大倍数，放大倍数误差
        self.z = queryZoom(ser=self.ser, address=self.address)
        self.z_last = 0
        self.ez = 0
        self.ez_last1 = 0
        self.ez_last2 = 0
        # pid控制的比例、积分、微分系数
        self.xkp = 10
        self.xki = 0
        self.xkd = 2
        self.ykp = 10
        self.yki = 0
        self.ykd = 2
        self.zoomkp = 400
        self.zoomki = 2000
        self.zoomkd = 0
        # 自动变焦情况下放大倍数理想值
        self.zoom = 0.8
        # 进程运行标志
        self.is_running = True
        # 追踪id
        self.confirm_id = -1
		# 程序状态标志
        self.zoom_factor = 0
        self.flag_zoom = True
        self.move = 'stop'

```

```
# 重要方法
run(self)：对摄像头的单次控制
reset(self)：清空之前的状态
```

#### VideoCapture() # 实现实时视频流的拉取

```
# 基本属性
		self.process, self.cap_info = ffmpeg_init(url) # 视频流对象和信息
        self.frame = None # 最新视频帧
        self.ret = None # 当前帧是否有效
        self.is_running = True # 进程运行标志
        self.size = self.getsize() # 视频帧的大小
        self.bytes = None # 存储视频帧的字节数组
        self.queue = queue # 进程间通信的队列
```

```
# 重要方法：
run(self)：拉取最新视频帧并发送给图像识别模块
```

