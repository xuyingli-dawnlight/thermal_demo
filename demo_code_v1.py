# this code is running on jetson nano for trasmitting data
import json
import platform
import sys
import time
import numpy
from queue import Queue
from enum import Enum
import paho.mqtt.client as mqtt
import numpy as np
from json import JSONEncoder
from uvctypes import *

libuvc = cdll.LoadLibrary("libuvc.so")

# settings for testing the mqtt send 
TASK_TOPIC = 'thermal_result'  # test topic to send
client_id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
client = mqtt.Client(client_id, transport='tcp')
client.connect("127.0.0.1", 1883, 60)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def clicent_main(message):

    """
    client send message 
    :param message: the main part of the message 
    :return:
    """
    time_now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    payload = {"msg": message}
    # publish(主题：Topic; 消息内容)
    client.publish(TASK_TOPIC, json.dumps(payload, ensure_ascii=False, cls=NumpyArrayEncoder))
    print("Successful send message!")
#
PT_USB_VID = 0x1e4e
PT_USB_PID = 0x0100

BUF_SIZE = 2
q = Queue(BUF_SIZE)
MAX_DATA_LENGTH = int(sys.argv[1])

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(frame.contents.height, frame.contents.width)
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return
    if not q.full():
        q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

class DataSource(Enum):
    VITAL_SIGNS = 1
    STATUS = 2
    ABNORMAL_EVENT = 3
    RAW_DATA = 4
    MEASUREMENT = 5

def main():

    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    data_idx, gap_idx, smooth_idx, collect = 0, 0, 0, False
    frame_gap = 9 * 120
    waitting_gap = 9 * 5
    smooth_length = 9 * 1000
    temp_queue = Queue(smooth_length)
    last_temp = 0

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(1)
    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)
        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                print("uvc_open error")
                exit(1)
            print("device opened!")
            print_device_info(devh)
            print_device_formats(devh)

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
            )

            set_manual_ffc(devh)
            print_shutter_info(devh)
            perform_manual_ffc(devh)

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)

            try:
                while True:
                    data = q.get()
                    if data is None:
                        break
                    data_idx = data_idx + 1
                    if data_idx % frame_gap == 0: # perform ffc regularlly
                        perform_manual_ffc(devh)
                        print_shutter_info(devh)
                        gap_idx = 0
                        smooth_idx = 0
                    gap_idx += 1
                    if gap_idx >= waitting_gap:
                        collect = True
                    else:
                        collect = False
                    if collect and smooth_idx < smooth_length:
                        # put data into temp
                        temp_queue.put((data - 27315) / 100.0)
                        smooth_idx += 1
                        
                    else:
                        temp_queue.queue.clear()
                    if not temp_queue.empty():
                        last_temp = temp_queue.queue
                        last_temp = list(last_temp)
                    clicent_main(last_temp)
                    temp_queue = Queue(smooth_length)
                    # print('{}\t{}\t{}'.format(time.time(), last_temp, smooth_idx))
                    # send the data to mqtt here
                    # clicent_main(last_temp)
                    
                    if data_idx >= MAX_DATA_LENGTH:
                        break
            finally:
                libuvc.uvc_stop_streaming(devh)
        finally:
            libuvc.uvc_unref_device(dev)
            
    finally:
        libuvc.uvc_exit(ctx)

if __name__ == '__main__':
    main()