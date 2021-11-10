FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ARG GID
ARG GNAME
ARG UNAME
ARG UID

WORKDIR .

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git

RUN pip install numpy
RUN pip install tqdm
RUN pip install scikit-learn
RUN pip install h5py
RUN pip install brevitas==0.6.0
RUN pip install onnxoptimizer==0.2.6
RUN pip install git+https://github.com/Xilinx/finn-base.git@feature/itu_competition_21#egg=finn-base[onnx]

# switch user
RUN groupadd -g $GID $GNAME
RUN useradd -M -u $UID $UNAME -g $GNAME
RUN usermod -aG sudo $UNAME
RUN chown -R $UNAME:$GNAME .
USER $UNAME