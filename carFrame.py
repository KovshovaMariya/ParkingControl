import sys
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
import time
import datetime
import sqlite3 as sl
from numpy import vstack
from mrcnn.model import MaskRCNN
from pathlib import Path
from threading import Thread
from imutils.video import FileVideoStream as Fvs


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.8


def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


ROOT_DIR = Path(".")

MODEL_DIR = ROOT_DIR / "logs"

COCO_MODEL_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

IMAGE_DIR = ROOT_DIR / "images"

model = MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                 config=MaskRCNNConfig())

model.load_weights(str(COCO_MODEL_PATH), by_name=True)

parked_car_boxes = None

video_capture = Fvs("rtsp://192.168.1.66:8080/h264_ulaw.sdp").start()
count = 1
time.sleep(1.0)
coords = []
new_cars = []
count_append, temp1, temp2 = 0, 0, 0

date_now = datetime.datetime.now()
con = sl.connect('parking-log.db')

while video_capture.more():
    frame = video_capture.read()
    if count % 60 == 0:
        frame = cv2.resize(frame, (720, 480))
        rgb_image = frame[:, :, ::-1]

        results = model.detect([rgb_image], verbose=0)
        r = results[0]

        if parked_car_boxes is None:
            parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        else:
            car_boxes = get_car_boxes(r['rois'], r['class_ids'])
            count_free_space = 0

            overlaps = mrcnn.utils.compute_overlaps(
                parked_car_boxes, car_boxes)

            for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
                count_change = count_free_space

                max_IoU_overlap = np.max(overlap_areas)

                y1, x1, y2, x2 = parking_area

                if max_IoU_overlap < 0.3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    count_free_space += 1

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    if count_free_space > 0:
                        count_free_space -= 1

                font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, f"Space aviable: " + f"{count_free_space}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.FILLED)

            if (temp1 != count_free_space) or (temp2 != len(parked_car_boxes)):
                date_now = datetime.datetime.now()
                sql = 'INSERT INTO LOG (datetime, message) values(?, ?)'
                data = [
                    (' ' + str(date_now), ' Свободно мест: ' +
                     str(count_free_space) + ' из ' + str(len(parked_car_boxes)))
                ]
                with con:
                    con.executemany(sql, data)

                temp1 = count_free_space
                temp2 = len(parked_car_boxes)

            count_len = 0
            count_len = len(car_boxes)

            if count_len+count_free_space > len(parked_car_boxes):
                new_cars.append(car_boxes[count_len - 1])
                count_append += 1
                is_new_car = False

                if count_append > 2:
                    start_i = count_append - 2
                else:
                    start_i = 0

                for i in range(start_i, len(new_cars) - 1):
                    if (new_cars[i][0] == new_cars[i+1][0]):
                        if (new_cars[i][1] == new_cars[i+1][1]):
                            if (new_cars[i][2] == new_cars[i+1][2]):
                                new_parked_cars = new_cars[i]
                                parked_car_boxes = vstack(
                                    [parked_car_boxes, new_parked_cars])
                                is_new_car = True

                if is_new_car:
                    print('TEST')
                    new_cars = []
                    count_append = 0
                    is_new_car = False

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    count += 1

video_capture.release()
cv2.destroyAllWindows()
