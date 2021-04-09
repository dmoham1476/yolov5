# gulfstream_yolov5

## Steps to bring up model server

docker run -it --runtime nvidia --net host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix  nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
git clone https://gecgithub01.walmart.com/d0m028p/gulfstream_yolov5.git
apt update
apt-get install python3-pip
pip3 install nano
#Comment out torch, torchvision, opencv-python in requirements.txt
pip3 install -r requirements.txt
apt install -y python3-opencv
#Download/copy the yolov5 trained model into the repo
python3 detect_api.py <path_to_model>
