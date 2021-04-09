# gulfstream_yolov5

## Steps to bring up model server on Xavier NX

Step 1:Run the jetpack NX pytorch base container\
docker run -it --runtime nvidia --net host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix  nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

Step 2:Clone the git repo\
git clone https://gecgithub01.walmart.com/d0m028p/gulfstream_yolov5.git

Step 3: Install dependancies\
apt update\
apt-get install python3-pip\
apt-get install nano\
nano requirements.txt\
#Comment out torch, torchvision, opencv-python in requirements.txt\
#Ctrl-O and then Ctrl-X in nano editor\
pip3 install -r requirements.txt\
apt install -y python3-opencv

Step 4: Start the yolov5 model server\
python3 detect_api.py ./yolov5_0330_best.pt
