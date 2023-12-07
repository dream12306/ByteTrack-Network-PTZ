import ffmpeg
import numpy as np
import cv2
import threading
import time

def ffmpeg_init(source):
    args = {
        "rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "flags": "low_delay"
    }    # 添加参数
    probe = ffmpeg.probe(source)
    cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    process1 = (
        ffmpeg
        .input(source, **args)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24',loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )
    return process1, cap_info





class VideoCapture:
    def __init__(self, url,queue):
        self.process, self.cap_info = ffmpeg_init(url)
        self.frame = None
        self.ret = None
        self.is_running = True
        self.size = self.getsize()
        self.bytes = None
        self.queue = queue

    def run(self):
        # print("开始执行")
        self.bytes = self.process.stdout.read(self.size[0] * self.size[1] * 3)  # 读取图片
        if not self.bytes:
            return
        self.frame = (
            np
            .frombuffer(self.bytes, np.uint8)
            .reshape([self.size[1], self.size[0], 3])
        )
        self.ret = self.frame is not None
        # in_frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
        if self.queue.full():
            self.queue.get()
        # print("已发送")
        self.queue.put({"ret_val": self.ret, "frame": self.frame})
        # cv2.imshow("123",self.frame)


    def run1(self,queue):
        self.queue = queue
        while self.is_running:
            self.bytes = self.process.stdout.read(self.size[0] * self.size[1] * 3)  # 读取图片
            if not self.bytes:
                continue
            self.frame = (
                np
                .frombuffer(self.bytes, np.uint8)
                .reshape([self.size[1], self.size[0], 3])
            )
            self.ret = self.frame is not  None
            # in_frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
            if queue.full():
                queue.get()
            queue.put({"ret_val":self.ret,"frame":self.frame})
            # cv2.imshow("123",self.frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

    def read(self):
        print("已读取")
        if not self.bytes:
            return None,None
        # 转成ndarray
        in_frame = (
            np
            .frombuffer(self.bytes, np.uint8)
            .reshape([self.size[1], self.size[0], 3])
        )
        # in_frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
        self.ret, self.frame = in_frame is not None, in_frame

    def getimage(self):
        return self.ret,self.frame

    def getimage1(self):
        in_bytes = self.process.stdout.read(self.size[0] * self.size[1] * 3)  # 读取图片
        if not in_bytes:
            return None,None
        # 转成ndarray
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([self.size[1], self.size[0], 3])
        )
        # in_frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
        self.ret, self.frame = in_frame is not None, in_frame
        return self.ret, self.frame

    def getsize(self):
        width = self.cap_info['width']  # 获取视频流的宽度
        height = self.cap_info['height']  # 获取视频流的高度
        return [width,height]

    def getfps(self):
        up, down = str(self.cap_info['r_frame_rate']).split('/')
        fps = eval(up) / eval(down)
        return fps




if __name__ == "__main__":
    rtsp_url1 = "rtsp://admin:abcd1234@192.168.1.108/mpeg4/ch1/sub/av_stream"
    rtsp = VideoCapture(rtsp_url1)
    time.sleep(2)
    while True:
        ret_cap,frame = rtsp.getimage1()
        print(ret_cap)
        if ret_cap:
            cv2.imshow("ffmpeg", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    rtsp.kill()             # 关闭
    cv2.destroyAllWindows()
