# https://realpython.com/blog/python/setting-up-a-simple-ocr-server/
# start with a base image
FROM centos:7

# install dependencies
RUN yum install -y which tar && yum install -y openssh-client openssh-server && yum install -y gcc gdb && yum install -y openmpi openmpi-devel && yum groupinstall -y "Development Tools"

# vvv no effect vvv why? vvv
RUN echo "module load mpi/openmpi-x86_64" >> /etc/rc.d/rc.local

# CMD ["python", "-c", "import cv2; print('cv2.__version__ = ' + str(cv2.__version__))"]
