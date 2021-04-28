import argparse
import time
from pathlib import Path
import base64


import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np
import sys

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadRS
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from flask import Flask, request, jsonify

def cover_detected_items(img, xywh, grey_color):
    (x,y,w,h) = xywh
    covered_img = img.copy()  #(shape h,w,c)
    bbox = np.array([[[grey_color]*3]*w]*h)
    #print("Jae - bbox", bbox.shape)
    covered_img[y-round(h/2-0.1):y+round(h/2+0.1), x-round(w/2-0.1):x+round(w/2+0.1),:] = bbox

    return covered_img

app = Flask(__name__)

weights = 'yolov5s.pt' if len(sys.argv) == 1 else sys.argv[1]
device_number = '' if len(sys.argv) <=2  else sys.argv[2]
device = select_device(device_number)
model = attempt_load(weights, map_location=device)  # load FP32 model

@app.route('/detect', methods=['GET', 'POST'])

def detect(save_img=False):
    save_img=False
    form_data = request.json
    source_type = form_data["source_type"]
    #np_list = form_data["numpy_list"]
    source = form_data['source']
    out = form_data['output']
    imgsz = form_data['imgsz']
    conf_thres = form_data['conf_thres']
    iou_thres = form_data['iou_thres']
    view_img = form_data['view_img']
    save_txt = form_data['save_txt']
    classes = form_data['classes']
    agnostic_nms = form_data['agnostic_nms']
    augment = form_data['augment']
    update = form_data['update']
    np_list = []
    #Camera specific metadata in the client api
    img_metadata  = form_data["image_list"]
    for img_table in img_metadata:
        np_list.append(img_table["numpy_list"][0])
    #source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    #webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #    ('rtsp://', 'rtmp://', 'http://'))

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
    depth_scale =0.0002500000118743628
    clipping_distance_in_meters = 0.98 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale



    webcam = False

    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    #device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    if source.lower().startswith('rs'):
        print("Hi Jae, source is 'rs'")
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadRS(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        print("***** Something wrong! Exit right away *****")
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    results = {}
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, depth_image in dataset:
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Results json
        results[path] = {}

        grey_color = 153
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # Write results
            covered_img = im0

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #bbox = det[:, :4].cpu().numpy()
                #print("Jae,  scaled det[:,:4] ", det[:,:4])
                #print("\nJae - bbox ",bbox.shape, bbox)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string
                    results[path][names[int(c)]] = n.item()

                for *xyxy, conf, cls in reversed(det):
                    """
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    """
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        # convert a list of float to a list of int
                        xywh = list(map(int,xywh))
                        #print("JAE: xywh", xywh)
                        covered_img = cover_detected_items(covered_img, xywh, grey_color)
                        #print(covered_img)
            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')




            # Remove background with 1m away
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, im0)
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, covered_img)

            print("testing.........", type(bg_removed))
            print(results)
            string = base64.b64encode(cv2.imencode('.jpg', bg_removed)[1]).decode()
            results["undetected_item"] = string
            return jsonify(results)
            #cv2.namedWindow("covered_img", cv2.WINDOW_NORMAL)
            #cv2.imshow("covered_img", covered_img)
            #cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            #cv2.imshow('RealSense', bg_removed)
            """
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    """

    # Stream results
    if view_img:
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
        cv2.imshow(str(p), bg_removed)
    print(type(bg_removed))
    print(f'Done. ({time.time() - t0:.3f}s)')
    return jsonify(bg_removed)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8000)
