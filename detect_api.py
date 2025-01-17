import argparse

import torch.backends.cudnn as cudnn

from flask import Flask, request, jsonify
from models.experimental import *
from utils.datasets import *
from utils.torch_utils import *
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from itemId import *
from collections import defaultdict

import sys

app = Flask(__name__)

weights = 'yolov5s.pt' if len(sys.argv) == 1 else sys.argv[1]
device_number = '' if len(sys.argv) <=2  else sys.argv[2]
device = select_device(device_number)

model = attempt_load(weights, map_location=device)  # load FP32 model

def consolidate(results):
    agg_list = []
    for cam, predict in results.items():
        agg_list.append(predict)

    union_results = defaultdict(set)
    #Aggregate all camera predictions
    for d in agg_list:
       	for k,v in d.items():
            union_results[k].add(v)
    #Assuming no false positives, keep maximum of all counts detected by each camera for an item
    max_dict = {label_to_itemId_map[k] : max(union_results[k]) for k in union_results}

    return max_dict

@app.route('/detect', methods=['GET', 'POST'])
def detect():
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

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    # Initialize
    # device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source_type, np_list, source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    results = {}
    dataset.mode = 'images'
    for path, img, im0s, vid_cap in dataset:
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
        #save_img = True

        # Results json
        results[path] = {}

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    #results[path][label_to_itemId_map[names[int(c)]]] = n.item()
                    results[path][names[int(c)]] = n.item()
                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    print("Testing.......", save_path)
                    cv2.imwrite(save_path + '.jpg', im0)
#                else:
#                    if vid_path != save_path:  # new video
#                        vid_path = save_path
#                        if isinstance(vid_writer, type(cv2.VideoWriter)):
#                            vid_writer.release()  # release previous video writer

#                        fourcc = 'mp4v'  # output video codec
#                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
#                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    return jsonify(consolidate(results))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 6000)
