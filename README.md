# gulfstream_yolov5

## Option 1 : Steps to bring up model server on Xavier NX using ngc pytorch container for Jetson

Step 1:Run the jetpack NX pytorch base container\
docker run -it --runtime nvidia --net host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix  nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

Step 2:Clone the git repo\
git clone https://gecgithub01.walmart.com/d0m028p/gulfstream_yolov5.git
cd gulfstream_yolov5\

Step 3: Install dependancies\
apt update\
apt-get install python3-pip\
apt-get install nano\
nano requirements.txt\
#Comment out opencv-python in requirements.txt\
#Ctrl-O and then Ctrl-X in nano editor\
pip3 install -r requirements.txt\
apt install -y python3-opencv\
pip3 install flask

Step 4: Start the yolov5 model server\
python3 detect_api.py ./yolov5_0330_best.pt

## Option 2 : Build from source without ngc container
sudo apt-get install python3-pip git\
sudo python3 -m pip -H uninstall torch\
sudo python3 -m pip -H uninstall torchvision\
sudo apt update\
sudo apt upgrade\
mkdir ~/gs_git && cd gs_git\
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl\
sudo apt-get install libopenblas-base libopenmpi-dev\
sudo pip3 install Cython\
sudo pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl\
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev\
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision\
cd torchvision\
export BUILD_VERSION=0.9.0\
sudo python3 setup.py install\
cd ~/gs_git\
git clone https://gecgithub01.walmart.com/d0m028p/gulfstream_yolov5.git
cd gulfstream_yolov5\
sudo pip3 install -U pip testresources\
pip3 install -r requirements.txt
Start the yolov5 model server\
python3 detect_api.py ./yolov5_0330_best.pt
