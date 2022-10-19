import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def extract_frames(cap, extract_ratio, useless_sum):
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extract_frames = int(n_total_frames * extract_ratio)
    index = 0
    stop = False
    for _ in tqdm(range(extract_frames)):
        ret, frame = cap.read()
        if np.sum(frame) == useless_sum:
            if stop is False:
                continue
            else:
                break
        else:
            stop = True
            cv2.imwrite(r'E:\xuxh\frames\{}.jpg'.format(index), frame)
            index += 1


def crop_frame(frame_path, cropped_path, size):
    dir = os.listdir(frame_path)
    for item in tqdm(dir):
        tmp_path = os.path.join(frame_path, item)
        img = cv2.imread(tmp_path)
        img_cropped = img[45: 1030, 300: 1620]
        img_resized = cv2.resize(img_cropped, size)
        cv2.imwrite(os.path.join(cropped_path, item), img_resized)


extract_ratio = 8 / 60
video_path = r'E:\xuxh\data\VS_Stim.avi'
frame_path = r'E:\xuxh\frames'
cropped_path = r'E:\xuxh\frames_cropped'
print(len(os.listdir(cropped_path)))
# size = (256, 256)
# crop_frame(frame_path, cropped_path, size)
# raw_frame = np.zeros(shape=(1080, 1920, 3))
# for i in range(10):
#     x = np.random.randint(0, len(os.listdir(frame_path)))
#     print(x)
#     tmp_frame = cv2.imread(frame_path + r'\{}.jpg'.format(x))
#     raw_frame += tmp_frame
# raw_frame /= 10
# raw_frame = raw_frame.astype(np.uint8)
# raw_frame = np.transpose(raw_frame, (2, 0, 1))
# single_frame = cv2.imread(frame_path + r'\200.jpg')
# trans_frame = np.transpose(single_frame, (2, 0, 1))
# img_cropped = single_frame[46: 1033, 290: 1620]
# size = (256, 256)
# img_resized = cv2.resize(img_cropped, size)
# plt.imshow(img_resized)
# plt.show(block=True)
# cap = cv2.VideoCapture(video_path)
# useless_sum = 716325120
# _, img = cap.read()
# cap.release()



