import requests
import json
import cv2
import os
import base64

json_file = 'api_data.json'
url = 'http://localhost:6000/detect'
with open(json_file) as f:
    data = json.load(f)
#    print(data)

#image_dir = '/home/deepa/work/yolov5/data/images'
image_dir = '/Users/d0m028p/image_work/gulfstream_yolov5/data/images'
img_list = []

i = 1
for file in os.listdir(image_dir):
    img_json  = {}
    img_json["metadata"] = {}
    img_json["numpy_list"] = []
    filepath = os.path.join(image_dir, file)
    image = cv2.imread(filepath)
    string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
#    img_list.append(string)
    img_json["numpy_list"].append(string)
#    print(img_json)
    img_json["metadata"]["id"] = i
    img_json["metadata"]["location"] = 'top'
    img_list.append(img_json)
    i += 1

data["image_list"] = img_list
#print(data)

server_return = requests.post(url, json=data)
server_return = server_return.json()

print(server_return["detected_items"])
jpg_original = base64.b64decode(server_return["undetected_item"])
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
img0 = cv2.imdecode(jpg_as_np, flags=1)

cv2.imwrite('color_img.jpg', img0)
cv2.imshow('Color image', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
