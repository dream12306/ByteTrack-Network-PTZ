import argparse
import os
import os.path as osp
import time
import cv2
import torch
import sys

sys.path.append("D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\BYTETRACK\\ByteTrack")
sys.path.append("D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\BYTETRACK\\ByteTrack\\tools")


from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import tkinter as tk
from PIL import Image, ImageTk
import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]



def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default=None, help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/mot/yolox_s_mix_det.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="pretrained/bytetrack_s_mot17.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        # img, ratio = preproc(img, self.test_size)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

window = None
boxes_tlwhs = []
confirm_id = -1
tp_ids = []
boxes_ids = []
offset_x = 0
offset_y = 0
unconfirmed_tp_ids = []
unconfirmed_tp_ids_state = []
predictor_myself = None
tracker_myself = None
timer_myself = None
move = ''
rtsp_url = "rtsp://admin:abcd1234@192.168.1.108/mpeg4/ch1/sub/av_stream"
tp_id_my = -1

from camera import Magnify
from ffmpeg_my import VideoCapture

flag_id = True
flag_zoom = True
zoom = None
zoom_int = 0

rtspcap = None








def magnify(boxes_ids, tp_ids, canvas,queue):
    global confirm_id
    global move
    global zoom_int
    global flag_zoom
    if not queue.full():
        queue.put({'confirm_id': confirm_id, 'boxes_ids': boxes_ids, 'boxes_tlwhs': boxes_tlwhs, 'move': move, 'zoom': zoom_int, 'flag_zoom': flag_zoom})
    else:
        queue.get()
        queue.put({'confirm_id': confirm_id, 'boxes_ids': boxes_ids, 'boxes_tlwhs': boxes_tlwhs, 'move': move, 'zoom': zoom_int, 'flag_zoom': flag_zoom})
        # print("queue full")







def update(vid_writer, current_time, args, results, frame_id, canvas, photo_image,exp,queue_mag,queue_rtsp):
    global boxes_tlwhs
    global boxes_ids
    global tp_ids
    global unconfirmed_tp_ids
    global unconfirmed_tp_ids_state
    global predictor_myself
    global tracker_myself
    global timer_myself
    global confirm_id
    judge = []
    if frame_id % 20 == 0:
        logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer_myself.average_time)))

    if not queue_rtsp.empty():
        data = queue_rtsp.get()
        ret_val = data['ret_val']
        frame = data['frame']
    else:
        print("queue->rtsp to main empty")
        ret_val = False
    if ret_val:
        if flag_id:
            outputs, img_info = predictor_myself.inference(frame, timer_myself)
            online_targets = tracker_myself.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                judge.append(tid)
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            boxes_tlwhs = copy.deepcopy(online_tlwhs)
            boxes_ids = copy.deepcopy(online_ids)

            if confirm_id>=0:
                if not confirm_id in boxes_ids:
                    confirm_id = -1

            n = 0
            for i, tp_id in enumerate(tp_ids):
                if (tp_id not in judge):
                    unconfirmed_tp_ids.append(tp_id)
                    unconfirmed_tp_ids_state.append(0)
                    del tp_ids[i - n]
                    n = n + 1
            n = 0
            for i, unconfirmed_tp_id in enumerate(unconfirmed_tp_ids):
                if (unconfirmed_tp_ids_state[i - n] < 20):
                    if (unconfirmed_tp_id in boxes_ids):
                        tp_ids.append(unconfirmed_tp_id)
                        del unconfirmed_tp_ids[i - n]
                        del unconfirmed_tp_ids_state[i - n]
                        n = n + 1
                    else:
                        unconfirmed_tp_ids_state[i - n] = unconfirmed_tp_ids_state[i - n] + 1
                else:
                    del unconfirmed_tp_ids[i - n]
                    del unconfirmed_tp_ids_state[i - n]
                    n = n + 1

            timer_myself.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      fps=1. / timer_myself.average_time, tp_ids=tp_ids)

            magnify(boxes_ids=boxes_ids,tp_ids=tp_ids,canvas=canvas,queue=queue_mag)
            img = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)  # 将图像转换为Image对象
            photo_image = ImageTk.PhotoImage(image=img)  # 将Image对象转换为PhotoImage对象
            canvas.create_image(0, 0, anchor='nw', image=photo_image)  # 在画布上显示图像
        else:
            magnify(boxes_ids=[], tp_ids=[], canvas=canvas, queue=queue_mag)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)  # 将图像转换为Image对象
            photo_image = ImageTk.PhotoImage(image=img)  # 将Image对象转换为PhotoImage对象
            canvas.create_image(0, 0, anchor='nw', image=photo_image)  # 在画布上显示图像

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            rtspcap.kill()  # 释放相机资源
            cv2.destroyAllWindows()  # 关闭所有窗口
    else:
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            rtspcap.kill()  # 释放相机资源
            cv2.destroyAllWindows()  # 关闭所有窗口
        print("get image failed")
    frame_id += 1
    window.after(20, update, vid_writer, current_time, args, results, frame_id, canvas, photo_image,exp,queue_mag,queue_rtsp)


def on_click_left(event):
    global boxes_tlwhs
    global boxes_ids
    global tp_ids
    global unconfirmed_tp_ids
    if event.num == 1:
        for i, box in enumerate(boxes_tlwhs):
            x1, y1 = box[0:2]
            x2 = x1 + box[2]
            y2 = y1 + box[3]
            if (x1 < event.x < x2 and y1 < event.y < y2):
                if boxes_ids[i] not in tp_ids and boxes_ids[i] not in unconfirmed_tp_ids:
                    tp_ids.append(boxes_ids[i])
                    break


def on_click_right(event):
    global boxes_tlwhs
    global boxes_ids
    global tp_ids
    global unconfirmed_tp_ids
    global unconfirmed_tp_ids_state
    if event.num == 3:
        for i, box in enumerate(boxes_tlwhs):
            x1, y1 = box[0:2]
            x2 = x1 + box[2]
            y2 = y1 + box[3]
            if (x1 < event.x < x2 and y1 < event.y < y2):
                if boxes_ids[i] in tp_ids:
                    tp_ids.remove(boxes_ids[i])
                    break
                elif boxes_ids[i] in unconfirmed_tp_ids:
                    del unconfirmed_tp_ids_state[unconfirmed_tp_ids.index(boxes_ids[i])]
                    unconfirmed_tp_ids.remove(boxes_ids[i])
                    break


def startid():
    global flag_id
    flag_id = True


def stopid():
    global flag_id
    global confirm_id
    confirm_id = -1
    flag_id = False


def autozoom():
    global flag_zoom
    flag_zoom = True


def starttrack():
    global confirm_id
    global tp_ids
    confirm_id = tp_ids[0]


def stoptrack():
    global confirm_id
    confirm_id = -1


def cancelselect():
    global tp_ids
    global confirm_id
    global unconfirmed_tp_ids
    global unconfirmed_tp_ids_state
    confirm_id = -1
    tp_ids = []
    unconfirmed_tp_ids = []
    unconfirmed_tp_ids_state = []

def up_press(event):
    global move
    global confirm_id
    confirm_id = -1
    move = 'up'


def down_press(event):
    global move
    global confirm_id
    confirm_id = -1
    move = 'down'


def left_press(event):
    global move
    global confirm_id
    confirm_id = -1
    move = 'left'


def right_press(event):
    global move
    global confirm_id
    confirm_id = -1
    move = 'right'


def release(event):
    global move
    global confirm_id
    confirm_id = -1
    move = 'stop'



def addzoom():
    global zoom
    global flag_zoom
    global zoom_int
    flag_zoom = False
    zoom_int = min(zoom_int+1,20)
    zoom.set(str(zoom_int))


def subtractzoom():
    global zoom
    global flag_zoom
    global zoom_int
    flag_zoom = False
    zoom_int = max(zoom_int-1,0)
    zoom.set(str(zoom_int))


def imageflow_demo(predictor, vis_folder, current_time, args,exp,queue_mag,size_screen,queue_rtsp):
    global window
    global predictor_myself
    global tracker_myself
    global timer_myself
    global zoom
    global zoom_int
    predictor_myself = predictor
    # cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    # cap = cv2.VideoCapture(rtsp_url)

    size = size_screen

    window = tk.Tk()  # 创建Tkinter窗口
    canvas = tk.Canvas(window, width=size[0], height=size[1]+300)  # 创建画布
    canvas.pack()
    canvas.bind("<Button-1>", on_click_left)  # 绑定鼠标点击事件到画布,并传递边界框
    canvas.bind("<Button-3>", on_click_right)  # 绑定鼠标点击事件到画布,并传递边界框

    print("canvas init success")

    i_up = tk.PhotoImage(file="D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\第二阶段\\多进程\\ByteTrack\\tools\\image\\up.png")
    i_down = tk.PhotoImage(file="D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\第二阶段\\多进程\\ByteTrack\\tools\\image\\down.png")
    i_left = tk.PhotoImage(file="D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\第二阶段\\多进程\\ByteTrack\\tools\\image\\left.png")
    i_right = tk.PhotoImage(file="D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\第二阶段\\多进程\\ByteTrack\\tools\\image\\right.png")
    i_add = tk.PhotoImage(file="D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\第二阶段\\多进程\\ByteTrack\\tools\\image\\+.png")
    i_subtract = tk.PhotoImage(file="D:\\1_dream_person\\小学期\\招商证券人工智能软件工程训练营\\第二阶段\\多进程\\ByteTrack\\tools\\image\\-.png")

    print("image read success")

    button_up = tk.Button(window, text='up', bg='white', image=i_up)
    button_down = tk.Button(window, text='down', bg='white', image=i_down)
    button_left = tk.Button(window, text='left', bg='white', image=i_left)
    button_right = tk.Button(window, text='right', bg='white', image=i_right)
    button_add = tk.Button(window, text='add', bg='white', image=i_add,command=addzoom)
    button_subtract = tk.Button(window, text='subtract', bg='white', image=i_subtract,command=subtractzoom)
    print("button of forward init success")

    zoom = tk.StringVar()
    zoom.set(str(zoom_int))
    text_zoom = tk.Label(window, textvariable=zoom,width=2, height=1, bg='white',fg='black',font=("Arial", 20))

    button_startid = tk.Button(window, text='开始识别', bg='white',command=startid)
    button_stopid = tk.Button(window, text='停止识别', bg='white',command=stopid)
    button_autozoom = tk.Button(window, text='自动变焦', bg='white',command=autozoom)
    button_starttrack = tk.Button(window, text='开始跟踪', bg='white',command=starttrack)
    button_stoptrack = tk.Button(window, text='取消跟踪', bg='white',command=stoptrack)
    button_cancelselect = tk.Button(window, text='取消选择', bg='white',command=cancelselect)

    print("button of work init success")

    button_up.place(x=40, y=size[1] + 50)
    button_down.place(x=40, y=size[1] + 50 + 80)
    button_left.place(x=0, y=size[1] + 50 + 40)
    button_right.place(x=80, y=size[1] + 50 + 40)
    button_add.place(x=0, y=size[1] + 50 + 120)
    button_subtract.place(x=80, y=size[1] + 50 + 120)
    text_zoom.place(x=42, y=size[1] + 50 + 121)

    button_up.bind("<ButtonPress-1>",up_press)
    button_up.bind("<ButtonRelease-1>", release)
    button_down.bind("<ButtonPress-1>", down_press)
    button_down.bind("<ButtonRelease-1>", release)
    button_left.bind("<ButtonPress-1>", left_press)
    button_left.bind("<ButtonRelease-1>", release)
    button_right.bind("<ButtonPress-1>", right_press)
    button_right.bind("<ButtonRelease-1>", release)

    button_startid.place(x=150, y=size[1] + 50 + 90)
    button_stopid.place(x=220, y=size[1] + 50 + 90)
    button_autozoom.place(x=290, y=size[1] + 50 + 90)
    button_starttrack.place(x=150, y=size[1] + 50 + 130)
    button_stoptrack.place(x=220, y=size[1] + 50 + 130)
    button_cancelselect.place(x=290, y=size[1] + 50 + 130)

    print("button init success")

    width = size[0]
    height = size[1]
    fps = 0
    save_folder = None
    save_path = None
    vid_writer = None
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, args.path.split("/")[-1])
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    tracker_myself = BYTETracker(args, frame_rate=60)
    timer_myself = Timer()
    frame_id = 0
    results = []
    photo_image = None
    update(vid_writer, current_time, args, results, frame_id, canvas, photo_image,exp,queue_mag,queue_rtsp)

    window.mainloop()  # 开始Tkinter主循环
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()  # 关闭所有窗口


def imageflow_demo1(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args,queue_mag,size_screen,queue_rtsp):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif (args.demo == "video" or args.demo == "webcam"):
        imageflow_demo(predictor, vis_folder, current_time, args,exp,queue_mag,size_screen,queue_rtsp)

from multiprocessing.managers import BaseManager
from multiprocessing import Process,Manager

if __name__ == "__main__":
    q_main_rtsp = Manager().Queue(2)
    BaseManager.register('VideoCapture', VideoCapture)
    BaseManager.register('Magnify', Magnify)
    manager = BaseManager()
    manager.start()
    rtsp = manager.VideoCapture("rtsp://admin:abcd1234@192.168.1.108/mpeg4/ch1/sub/av_stream")
    time.sleep(2)
    pro_rtsp = Process(target=rtsp.run, args=(q_main_rtsp,))

    size = rtsp.getsize()

    # 创建队列
    q_main_mag = Manager().Queue(2)

    mag = manager.Magnify(size=size)
    pro_mag = Process(target=mag.run, args=(q_main_mag,))

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    # main(exp, args)
    processmain = Process(target=main, args=(exp, args, q_main_mag,size,q_main_rtsp))
    pro_rtsp.start()
    pro_mag.start()
    processmain.start()
    processmain.join()
    pro_rtsp.join()
    #pro_mag.join()