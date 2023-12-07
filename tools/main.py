from multiprocessing.managers import BaseManager
from multiprocessing import Process, Manager, Pool

from camera import Magnify
from ffmpeg_my import VideoCapture
import time
from demo_track import make_parser, get_exp, main
import cv2


def run_mag(magnify):
    # print("mag run")
    magnify.run()


def run_rtsp(rtsp):
    # print("rtsp run")
    rtsp.run()


def exit(event):
    global flag_exit
    flag_exit = 1

if __name__ == "__main__":
    flag_exit = 0
    q_main_rtsp = Manager().Queue(2)
    q_main_mag = Manager().Queue(2)
    BaseManager.register('VideoCapture', VideoCapture)
    BaseManager.register('Magnify', Magnify)
    manager = BaseManager()
    manager.start()

    rtsp = manager.VideoCapture("rtsp://admin:abcd1234@192.168.1.108/mpeg4/ch1/sub/av_stream", q_main_rtsp)
    time.sleep(2)
    size = rtsp.getsize()
    print(size)
    print("rtsp init success")
    mag = manager.Magnify(size=size, queue=q_main_mag)
    print("mag init success")

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    pool = Pool(processes=3)
    print("pool init success")
    pool.apply_async(func=main, args=(exp, args, q_main_mag, size, q_main_rtsp),callback=exit)
    print(1)
    r_mag = pool.apply_async(func=run_mag, args=(mag,))
    print(2)
    r_rtsp = pool.apply_async(func=run_rtsp, args=(rtsp,))
    print(3)
    while not flag_exit:

        if (r_mag.ready()):
            # print("mag ready")
            r_mag = pool.apply_async(func=run_mag, args=(mag,))
        else:
            # print("mag not ready")
            pass
        if (r_rtsp.ready()):
            # print("rtsp ready")

            r_rtsp = pool.apply_async(func=run_rtsp, args=(rtsp,))
        else:
            # print("rtsp not ready")
            pass
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    print("已退出")
    pool.close()
    pool.join()
    mag.reset()
    print("-----------------")
    print("执行完毕")
    # processmain.join()
    # pro_mag.join()
