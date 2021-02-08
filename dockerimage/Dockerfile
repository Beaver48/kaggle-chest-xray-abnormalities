FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV LANG C.UTF-8

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update &&\
    apt-get install -y software-properties-common \
                       build-essential \
                       cmake \
                       git \
                       ca-certificates \
                       wget \
                       vim \
                       yasm \
                       telnet \
                       curl \
                       zip \
                       nodejs \
                       npm \
                       libicu-dev \
                       zlib1g-dev \
                       libsnappy-dev \
                       unzip \
                       pkg-config \
                       libblas-dev \
                       liblapack-dev \
                       graphviz \
                       libhdf5-dev \
                       swig \
                       python3-distutils &&\
    add-apt-repository -y ppa:deadsnakes/ppa &&\
    rm -rf /var/lib/apt/lists/*

ARG PYTHON=python3

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py pip==21.0 &&\
    rm get-pip.py &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 --no-cache-dir install numpy==1.19.1


RUN apt-get update &&\
    apt-get install -y pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-bad1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-dev \
        libprotobuf-dev \
        pkg-config \
        protobuf-compiler \
        python-yaml \
        python-six \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION="4.4.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& curl -L https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -o opencv_contrib-${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& unzip opencv_contrib-${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake  -D WITH_CUDA=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
        -D WITH_GSTREAMER=ON \
        -D WITH_LAPACK=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_ml=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D PYTHON_EXECUTABLE=$(which python3) \
        -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make -j "$(nproc)" install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV FORCE_CUDA="1"

RUN pip install --no-cache-dir torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone --branch 'v1.2.6' https://github.com/open-mmlab/mmcv.git /mmcv \
&& cd /mmcv && MMCV_WITH_OPS=1 pip install -e .

RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .


COPY requirements.txt /tmp/

RUN pip --no-cache-dir install -r /tmp/requirements.txt

RUN sed -i 's/\.jpg/\.png/g' /mmdetection/mmdet/datasets/xml_style.py

WORKDIR /workdir
