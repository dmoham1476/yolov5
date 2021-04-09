import requests
import json
import cv2
import os
import base64

json_file = 'api_data.json'
url = '192.168.1.5:6000/detect'
with open(json_file) as f:
    data = json.load(f)
    print(data)

#image_dir = '/home/deepa/work/yolov5/data/images'
image_dir = '/Users/d0m028p/gulfstream_yolov5/data/images'
img_list = []
for file in os.listdir(image_dir):
    filepath = os.path.join(image_dir, file)
    image = cv2.imread(filepath)
    string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
    img_list.append(string)
data["numpy_list"] = img_list

server_return = requests.post(url, json=data)
print(server_return.text)
