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
print(server_return[1])
jpg_original = base64.b64decode(server_return["undetected_item"])
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)

cv2.imwrite('color_img.jpg', jpg_as_np)
cv2.imshow('Color image', jpg_as_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
