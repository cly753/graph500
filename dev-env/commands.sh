#!/bin/sh

exit(0)

# build the image
# save to repo cly753
# as name openmpi
# with not tag
# use the Dockerfile from . (current dir)
# and Remove intermediate containers after a successful build
docker build --rm -t cly753/openmpi .

# run (create and start) the container
# from repo cly753
# with name openmpi
# with not tag
# and mount /Users/cly/Dropbox/code/isc/graph500/graph500-2.1.4 to container /graph500-2.1.4
docker run -it -v /Users/cly/Dropbox/code/isc/graph500/graph500-2.1.4:/graph500-2.1.4 cly753/openmpi

# open a bash in the running container
# docker exec -it [CONTAINER] bash

# show the docker-machine ip
# named default
docker-machine ip default

# push the image to docker hub
docker push cly753/openmpi

# for centos
module load mpi/openmpi-x86_64
