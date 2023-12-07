import serial
import cv2
import time


def setspeed(speed,ser,address,mode='x'):
    if mode == 'x':
        if speed > 0:
            code = 0x02
        else:
            code = 0x04
        speed = abs(speed)
        if speed > 0xffff:
            speed = 0xffff
        elif speed < 0:
            speed = 0
    elif mode == 'y':
        if speed > 0:
            code = 0x08
        else:
            code = 0x10
        speed = abs(speed)
        if speed > 0xffff:
            speed = 0xffff
        elif speed < 0:
            speed = 0
    else:
        return
    last_four = speed & 0xFFFF
    # 分开前两位和后两位
    first_two = (last_four & 0xFF00) >> 8
    last_two = last_four & 0x00FF
    cmd = [0xff, 0x01, 0x00, code, first_two, last_two]  # 控制位置：0x4B
    cmd = cmd + [(sum(cmd)+1) % 256]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    # response = ser.read(7)  # 返回的数据长度应为7字节

def setspeedxy(speedx,speedy,ser,address):
    if speedx>=0 and speedy>=0:
        code = 0x0A
    elif speedx>=0 and speedy<0:
        code = 0x12
    elif speedx<0 and speedy>=0:
        code = 0x0C
    elif speedx<0 and speedy<0:
        code = 0x14
    else:
        return
    last_fourx = abs(speedx) & 0xFFFF
    last_foury = abs(speedy) & 0xFFFF
    # 分开前两位和后两位
    first_two = min((last_fourx & 0xFF00) >> 8,0x3F)
    last_two = min((last_foury & 0xFF00) >> 8,0x3F)
    print('---------------------')
    print(first_two)
    print(last_two)
    print('---------------------')
    cmd = [0xff, 0x01, 0x00, code, first_two, last_two]  # 控制位置：0x4B
    cmd = cmd + [(sum(cmd)+1) % 256]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    # response = ser.read(7)  # 返回的数据长度应为7字节





def setstop(ser,address):
    print("stop run!!!!!!!!!!!")
    cmd = [0xff, 0x01, 0x00, 0x00, 0x00, 0x00]
    cmd = cmd + [sum(cmd) % 256 + 1]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    # 读取返回的数据
    response = ser.read(7)  # 返回的数据长度应为7字节



def moveto(position, ser, address, mode='x'):
    if (mode == 'x'):
        code = 0x4B
        if (position > 35999):
            position = 35999
        if (position < 0):
            position = 0
    elif (mode == 'y'):
        code = 0x4D
        if (position > 9999):
            position = 9999
        if (position < 0):
            position = 0
    else:
        return
    last_four = position & 0xFFFF
    # 分开前两位和后两位
    first_two = (last_four & 0xFF00) >> 8
    last_two = last_four & 0x00FF
    cmd = [0xff, 0x01, 0x00, code, first_two, last_two]  # 控制位置：0x4B
    cmd = cmd + [sum(cmd) % 256 + 1]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    response = ser.read(7)  # 返回的数据长度应为7字节


def queryPosition(ser, address, mode='x'):
    if mode == 'x':
        cmd = [0xff, 0x01, 0x00, 0x51, 0x00, 0x00]  # 查询x位置
    elif mode == 'y':
        cmd = [0xff, 0x01, 0x00, 0x53, 0x00, 0x00]  # 查询y位置
    else:
        return False
    cmd = cmd + [(sum(cmd) + 1) % 256]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    # 读取返回的数据
    response = ser.read(7)  # 返回的数据长度应为7字节
    # 解析返回的数据
    time.sleep(0.005)
    try:
        pan_position = (response[4] << 8) | response[5]  # 平移位置在返回数据的第5和第6字节
    except:
        return False
    return pan_position


def setZoom(ser, address, zoom):
    if (zoom > 0xFFFF / 4):
        zoom = 0xFFFF / 4
    if (zoom < 0):
        zoom = 0

    last_four = zoom & 0xFFFF
    # 分开前两位和后两位
    first_two = (last_four & 0xFF00) >> 8
    last_two = last_four & 0x00FF
    cmd = [0xff, 0x01, 0x00, 0x4F, first_two, last_two]  # 控制位置：0x4B
    cmd = cmd + [(sum(cmd)+1) % 256]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    # response = ser.read(7)  # 返回的数据长度应为7字节


def queryZoom(ser, address):
    cmd = [0xff, 0x01, 0x00, 0x55, 0x00, 0x00]
    cmd = cmd + [sum(cmd) % 256 + 1]
    crc = 0x00  # CRC 校验
    # 计算 CRC 校验值
    for b in cmd:
        crc ^= b
    # 发送控制指令
    ser.write(bytes([address] + cmd + [crc]))
    # 读取返回的数据
    response = ser.read(7)  # 返回的数据长度应为7字节
    time.sleep(0.005)
    # 解析返回的数据
    try:
        Zoom = (response[4] << 8) | response[5]  # 平移位置在返回数据的第5和第6字节
    except:
        return False
    return Zoom


from multiprocessing import Process

class Magnify():
    def __init__(self, size,queue, *args, **kwargs):
        canvas_width = size[0]
        canvas_height = size[1]

        self.queue = queue
        # 串口配置
        self.size = size
        self.ser = serial.Serial('com4', baudrate=19200, timeout=0.5)
        self.address = 1  # 设备地址
        setZoom(self.ser,self.address,0)
        setstop(self.ser, self.address)
        self.canvas_height = int(canvas_height)
        self.canvas_width = int(canvas_width)
        self.canvas_center_x = int(canvas_width / 2)
        self.canvas_center_y = int(canvas_height / 2)

        self.speedx = 0
        self.speedx_last = 0
        self.ex = 0
        self.ex_last1 = 0
        self.ex_last2 = 0

        self.speedy = 0
        self.speedy_last = 0
        self.ey = 0
        self.ey_last1 = 0
        self.ey_last2 = 0
        self.z = queryZoom(ser=self.ser, address=self.address)
        self.z_last = 0
        self.ez = 0
        self.ez_last1 = 0
        self.ez_last2 = 0
        self.xkp = 10
        self.xki = 0
        self.xkd = 2
        self.ykp = 15
        self.yki = 0
        self.ykd = 3
        self.zoomkp = 400
        self.zoomki = 2000
        self.zoomkd = 0
        self.zoom = 0.8
        self.is_running = True
        self.i = 0
        self.confirm_id = -1
        self.tp_tlwh = [0,0,self.canvas_width,self.canvas_height]

        self.zoom_factor = 0
        self.flag_zoom = True
        self.move = 'stop'
        # 创建并启动进程
        # self.thread = multiprocessing.Process(target=self.run)
        # self.thread.start()


    def get_i(self):
        return self.i

    def run2(self):
        print(1234567895348967)
        print(self.i)
        self.i += 1
        print(self.get_i())
        return self

    def run(self):
        flag = 0
        if not self.queue.empty():
            data = self.queue.get()
            zoom_factor = data['zoom']
            self.flag_zoom = data['flag_zoom']
            move = data['move']
            confirm_id = data['confirm_id']
            boxes_ids = data['boxes_ids']
            boxes_tlwhs= data['boxes_tlwhs']
            # print('------------------')
            # print(zoom_factor)
            # print(self.zoom_factor)
            # print(self.flag_zoom)
            # print('-----------------')
            if confirm_id >= 0:

                if confirm_id in boxes_ids:
                    index = boxes_ids.index(confirm_id)
                    self.tp_tlwh = boxes_tlwhs[index]
                    self.confirm_id = confirm_id
                    flag = 1
            else:
                if move == 'left':
                    setspeed(-8224,self.ser,self.address,'x')
                    self.move = 'move'
                elif move == 'right':
                    setspeed(8224,self.ser,self.address,'x')
                    self.move = 'move'
                elif move == 'up':
                    setspeed(-8224,self.ser,self.address,'y')
                    self.move = 'move'
                elif move == 'down':
                    setspeed(8224,self.ser,self.address,'y')
                    self.move = 'move'
                elif move == 'stop':
                    if self.move != 'stop':
                        setstop(self.ser,self.address)
                        self.move = 'stop'
                if self.confirm_id != -1:
                    setstop(self.ser,self.address)
                    self.confirm_id = -1
            if self.zoom_factor != zoom_factor and not self.flag_zoom:
                # print('+++++++++++++++++')
                # print(int(zoom_factor * 0xffff / 4 / 20))
                setZoom(self.ser,self.address,zoom_factor * int(0xffff / 4 / 20))
                # print('+++++++++++++++++------------')
                self.zoom_factor = zoom_factor
                time.sleep(0.01)

        if flag:
            x1, y1, w, h = self.tp_tlwh
            x2 = x1 + w
            y2 = y1 + h
            if y1 < 0:
                y1 = 0
                y2 = min(self.canvas_height, y2)
            if y2 > self.canvas_height:
                y2 = self.canvas_height
                y1 = max(0, y1)
            if x1 < 0:
                x1 = 0
                x2 = min(self.canvas_width, x2)
            if x2 > self.canvas_width:
                x2 = self.canvas_width
                x1 = max(0, x1)

            # 要放大的区域的中心坐标
            target_center_x = (x1 + x2) / 2
            target_center_y = (y1 + y2) / 2
            # 要放大的区域的尺寸
            target_width = x2 - x1
            target_height = y2 - y1

            self.z_last = self.z
            self.ez_last2 = self.ez_last1
            self.ez_last1 = self.ez
            self.ez = self.zoom - max(target_width / self.canvas_width, target_height / self.canvas_height)

            self.speedx_last = self.speedx
            self.ex_last2 = self.ex_last1
            self.ex_last1 = self.ex
            self.ex = -1 * (target_center_x - self.canvas_center_x)

            self.speedy_last = self.speedy
            self.ey_last2 = self.ey_last1
            self.ey_last1 = self.ey
            self.ey = 1 * (target_center_y - self.canvas_center_y)

            self.z = int(self.z_last + (self.zoomkp + self.zoomki + self.zoomkd) * self.ez - \
                         (self.zoomkp + 2 * self.zoomkd) * self.ez_last1 + self.zoomkd * self.ez_last2)

            self.speedx = int(self.speedx_last + (self.xkp + self.xki + self.xkd) * self.ex - \
                              (self.xkp + 2 * self.xkd) * self.ex_last1 + self.xkd * self.ex_last2)

            self.speedy = int(self.speedy_last + (self.ykp + self.yki + self.ykd) * self.ey - \
                              (self.ykp + 2 * self.ykd) * self.ey_last1 + self.ykd * self.ey_last2)
            if self.z < 0:
                self.z = 0
            elif self.z > 0xFFFF / 4:
                self.z = 0xFFFF / 4

            if self.speedx < -1 * 0x3FFF:
                self.speedx = -1 * 0x3FFF
            elif self.speedx > 1 * 0x3FFF:
                self.speedx = 1 * 0x3FFF

            if self.speedy < -1 * 0x3FFF:
                self.speedy = 1 * 0x3FFF
            elif self.speedy > 1 * 0x3FFF:
                self.speedy = 1 * 0x3FFF
            print('-------------------------')
            print('x')
            print(self.speedx)
            print('y')
            print(self.speedy)
            print('z')
            print(self.z)
            print(self.flag_zoom)
            print('-------------------------')

            if self.flag_zoom:

                setZoom(zoom=self.z, ser=self.ser, address=self.address)
                time.sleep(0.01)
            if abs(self.speedx) >0xFF and abs(self.speedy)>0xFF:
                setspeedxy(self.speedx, self.speedy, self.ser, self.address)
            elif abs(self.speedx) > 0xFF and abs(self.speedy)<=0xFF:
                setspeed(self.speedx, self.ser, self.address)
            elif abs(self.speedx) <= 0xFF and abs(self.speedy) > 0xFF:
                setspeed(self.speedy, self.ser, self.address)
            else:
                setstop(self.ser,self.address)



    def stop(self):
        self.is_running = False
        # self.thread.terminate()

    def reset(self):
        setZoom(self.ser,self.address,0)
        setstop(self.ser,self.address)
        self.speedx = 0
        self.speedx_last = 0
        self.ex = 0
        self.ex_last1 = 0
        self.ex_last2 = 0
        self.speedy = 0
        self.speedy_last = 0
        self.ey = 0
        self.ey_last1 = 0
        self.ey_last2 = 0
        self.z = queryZoom(ser=self.ser, address=self.address)
        self.z_last = 0
        self.ez = 0
        self.ez_last1 = 0
        self.ez_last2 = 0
        self.zoom = 0.8
        self.is_running = True
        self.i = 0
        self.confirm_id = -1
        self.tp_tlwh = [0, 0, self.canvas_width, self.canvas_height]


if __name__ == "__main__":
    # from multiprocessing.managers import BaseManager
    # from multiprocessing import Process, Manager
    # BaseManager.register('Magnify', Magnify)
    # manager = BaseManager()
    # manager.start()
    # mag = manager.Magnify(size=[0,0])
    # mag = Magnify(size=[0,0])
    ser = serial.Serial('com4', baudrate=19200, timeout=0.5)
    address = 1
    # setspeedxy(2000,2000,ser,address)
    setstop(ser,address)
    # setspeed(2024,ser,address,'y')
    # time.sleep(0.001)
    # setstop(ser,address)
    # setspeed(-10024, ser, address, 'x')
    # time.sleep(1)
    # setstop(ser,address)
    # print(queryPosition(ser, address, 'x'))

    print(queryPosition(ser, address, 'y'))
