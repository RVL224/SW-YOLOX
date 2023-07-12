#docker build -t shift:latest .

# Startup sample
#mkdir -p pretrained && \
#mkdir -p YOLOX_outputs && \
xhost +local: && \
sudo docker run --gpus all -it --rm \
-v $PWD/:/workspace/SW-YOLOX/ \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
shift:latest

#enter the following code after entering the container
#sudo python3 setup.py develop
