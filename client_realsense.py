import requests
import json
import cv2
import os
import base64

json_file = 'api_data_camera.json'
url = 'http://localhost:8000/detect'
with open(json_file) as f:
    data = json.load(f)

server_return = requests.post(url, json=data)
print(server_return.text)
