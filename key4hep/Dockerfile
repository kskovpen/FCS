FROM ubuntu:latest as base
FROM base as builder

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y git make wget lsb-release dpkg
#RUN apt-get install -y git make wget lsb-release libncurses5 python3 python3-pip python3-distutils virtualenv dpkg gcc-11 fuse
#libgfortran5 cpio
#RUN gcc --version

#RUN python3 -m pip install --upgrade pip
#RUN python3 -m pip install pyhf deepdiff kaleido plotly matplotlib pydash jax jaxlib torch torchvision
#RUN python3 -m pip install --upgrade jax jaxlib

RUN git clone https://github.com/spack/spack
RUN git clone https://github.com/key4hep/key4hep-spack

RUN wget https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb
RUN dpkg -i cvmfs-release-latest_all.deb
RUN rm -f cvmfs-release-latest_all.deb
RUN apt-get update
RUN apt-get install -y cvmfs

COPY key4hep/install.sh .

ENTRYPOINT ["/bin/bash", "install.sh"]
