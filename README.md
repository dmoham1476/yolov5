# gulfstream_yolov5

## Steps to bring up model server on Xavier NX

Step 1:Run the jetpack NX pytorch base container\
docker run -it --runtime nvidia --net host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix  nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3\

Step 2:Clone the git repo\
git clone https://gecgithub01.walmart.com/d0m028p/gulfstream_yolov5.git \

Step 3: Install dependancies\
apt update\
apt-get install python3-pip\
pip3 install nano\
#Comment out torch, torchvision, opencv-python in requirements.txt\
pip3 install -r requirements.txt\
apt install -y python3-opencv\

Step 4: Download trained model weights\
#Download/copy the yolov5 trained model into the repo\

Step 5: Start the yolov5 model server\
python3 detect_api.py <path_to_model>
