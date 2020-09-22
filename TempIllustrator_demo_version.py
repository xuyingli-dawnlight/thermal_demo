import sys
sys.path.insert(1, '/Users/xuyingli/Desktop/research/yolov3')
from models import *  # set ONNX_EXPORT in models.p
from PIL import Image, ImageStat
import json
import time
from queue import Queue
from threading import Thread
import cv2
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
import numpy
from queue import Queue
import scipy.signal as signal

#signal filter settings
_filter_params ={'high_cut_off': 0.15, 'low_cut_off': 0.5, 'sample_rate': 559/30, 'order': 3} 
def butter_filter(_x, _type, _param):
    _sample_rate = _param['sample_rate']
    _cutoff_hz, order = _param['cut_off'], _param['order']
    _nyq_rate = 0.5 * _sample_rate
    _normal_cutoff = _cutoff_hz / _nyq_rate
    b, a = signal.butter(order, _normal_cutoff, btype=_type, analog=False)
    y = signal.filtfilt(b, a, _x)
    return y

def bandpass_filters(_x, _filter_params):
    _filter_params['cut_off'] = np.array([_filter_params['high_cut_off'],
                                          _filter_params['low_cut_off']])
    return butter_filter(_x, 'bandpass', _filter_params)


# model settings here
weights = '/Users/xuyingli/Desktop/research/yolov3/weights/best_400.pt'
cfg = '/Users/xuyingli/Desktop/research/yolov3/cfg/yolov3-face.cfg'
names = load_classes('/Users/xuyingli/Desktop/research/yolov3/data/coco.names')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
imgsz = 512
device = torch_utils.select_device(device='cpu')
model = Darknet(cfg, imgsz)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
# Eval mode
model.to(device).eval()

half = True and device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()
#

font = {
    'family': 'serif',
    'color': 'darkred',
    'weight': 'normal',
    'size': 28,
}

def cal_mean_stats(x,cv_frame):
    # find nose area
    x_0 = int(x[0] + 2 * (x[2] - x[0]) / 5)
    x_1 = int(x[1] + (x[3] - x[1]) / 2)
    x_2 = x_0 + int(3 * (x[2] - x[0]) / 16)
    x_3 = x_1 + int(1 * (x[3] - x[1]) / 4)
    cropped = cv_frame[x_0:x_2, x_1:x_3]
    # calculate mean value
    return np.mean(cropped)


def draw(lst):
    print(lst)
    plt.plot(lst)
    plt.show()

def Convert(string): 
    li = list(string.split(" ")) 
    return li 

# result of the mean pixel
result = []
length = 300

class TempDetectionIllustrator(object):

    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.topics = ['thermal_result']
        self.client.connect("192.168.55.1", 1883, 60)
        self.data_queue, self.data_idx = Queue(), 0

    def process(self):
        while self.data_queue.empty():
            time.sleep(0.1)

        msg = self.data_queue.get()
        # initialize figure
        fig, ax = plt.subplots(1, 2)
        # img data
        frame = np.array(msg['frame'], dtype=np.uint8)
        faces = np.array(msg['data']) if 'faces' in msg else np.array([])
        img_fig = ax[0].imshow(frame)
        hist_data = []
        for face in faces:
            rect = patches.Rectangle((face[0], face[1]), face[2], face[3], linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
            hist_data.extend(np.array(frame)[face[0]: face[0] + face[2], face[1]: face[1] + face[3]].flatten())
        ax[1].hist(hist_data, bins=list(range(0, 256)))

        def animate(_):
            ax[0].cla()
            t_msg = self.data_queue.get()
            t_frame = np.array(t_msg['frame'], dtype=np.uint8)
            ax[0].imshow(t_frame)
            t_faces = np.array(t_msg['faces']) if 'faces' in t_msg else np.array([])
            t_hist_data = []
            for t_face in t_faces:
                rect = patches.Rectangle((t_face[0], t_face[1]), t_face[2], t_face[3], linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax[0].add_patch(rect)
                t_hist_data.extend(
                    np.array(t_frame)[t_face[0]: t_face[0] + t_face[2], t_face[1]: t_face[1] + t_face[3]].flatten())
            ax[1].cla()
            ax[1].hist(t_hist_data)
            return img_fig,

        _ = animation.FuncAnimation(fig, animate, interval=200, blit=True)
        plt.show()

    def process_by_cv(self):
        factor = 5
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)

        def animate_cv(i):
            x = np.linspace(0, 1, 500)
            line.set_data(x,numpy.array(lst[i])) 
            return line
            
        plt.ion()
        while True:
            if self.data_queue.empty():
                time.sleep(0.1)
                continue
            msg = self.data_queue.get()
            np_frame = np.array(msg, dtype=np.uint8)
            cv_frame = cv2.resize(np_frame, (512, 512))
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_GRAY2RGB)
            cv_frame = cv2.normalize(cv_frame, cv_frame, 0, 255, cv2.NORM_MINMAX)
            
            # get the face here use yolov3
            # then use the rectangle to get the face
            # use cv2 for testing
            face_cascade = cv2.CascadeClassifier('/Users/xuyingli/Desktop/research/sensor-analysis-master/src/thermal_temp/untitled.xml')
            faces = face_cascade.detectMultiScale(cv_frame, 1.3, 5)

            '''
            # user re-trained yolo model
            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            img = torch.zeros((1, 3, 512, 512), device=device) 
            img = torch.from_numpy(cv_frame).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                 img = img.unsqueeze(0)
            img = img.permute(0,3,1,2) 
            # Inference
            pred = model(img, augment=True)[0]
            pred = non_max_suppression(pred, 0.3, 0.6,
                                   multi_label=False, agnostic=True)
            
            for i, det in enumerate(pred):
                gn = torch.tensor(cv_frame.shape)[[1, 0, 1, 0]] 

                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_frame.shape).round()
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, cv_frame, label=label, color=colors[int(cls)])
            '''
            
            if len(faces) != 0:
                for (x,y,w,h) in faces:
                    # cv2.rectangle(cv_frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    xyxy = (x,y,x+w,y+1.15*h)
                    label = '%s %.2f' % (names[int(0)], 1)
                    x = cal_mean_stats(xyxy,cv_frame)
                    plot_one_box(xyxy, cv_frame, label=label, color=colors[int(0)])
                    print(x)
                    if len(result) >= 50:
                        # draw the image
                        y = bandpass_filters(result, _filter_params)
                        plt.pause(0.01)
                        # 清除上一次显示
                        plt.cla()
                        plt.plot(y)
                        # update the graph
                        result.pop(5)
                    else:
                        result.append(x)


            cv2.imshow("thermal", cv_frame)
            cv2.waitKey(1)

    def on_connect(self, _client, _user_data, _flags, _rc):
        print('Connected with result code: {}, {}'.format(_rc, _client))
        print('Subscribing to {}...'.format('thermal_result'))
        self.client.subscribe('thermal_result')

    @staticmethod
    def on_subscribe(_client, _user_data, _mid, _granted_qos):
        print('Subscribed result: {}, QoS: {}'.format(_user_data, _granted_qos))

    def on_message(self, _client, _user_data, _msg):
       
        msg_json = json.loads(_msg.payload)
        
        for instance in msg_json['msg']:
            self.data_queue.put(instance)

    def run(self):
        self.client.loop_forever()


if __name__ == '__main__':
    app = TempDetectionIllustrator()
    subscribe_thread = Thread(target=app.run)
    subscribe_thread.start()
    app.process_by_cv()
