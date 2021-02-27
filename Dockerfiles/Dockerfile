FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER itaya
ARG DEBIAN_FRONTEND="noninteractive"

ENV PYENV_ROOT="/opt/pyenv" \
    PATH="/opt/pyenv/bin:/opt/pyenv/shims:$PATH" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8"

RUN apt-get update && apt-get install -y --no-install-recommends \
        # ubuntu16.04 -> libssl-dev
        # ubuntu18.04 -> libssl1.0-dev
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        htop \
        libbz2-dev \
        libffi-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libopencv-dev \
        libprotobuf-dev \
        libreadline-dev \
        libsnappy-dev \
        libsqlite3-dev \
        # libssl1.0-dev \
        libssl-dev \
        llvm \
        make \
        netbase \
        pkg-config \
        protobuf-compiler \
        screen \
        tk-dev \
        tmux \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyenv-version.txt python-versions.txt /

RUN git clone -b `cat /pyenv-version.txt` --single-branch --depth 1 https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && for version in `cat /python-versions.txt`; do pyenv install $version; done \
    && pyenv global `cat /python-versions.txt` \
    && find $PYENV_ROOT/versions -type d '(' -name '__pycache__' -o -name 'test' -o -name 'tests' ')' -exec rm -rf '{}' + \
    && find $PYENV_ROOT/versions -type f '(' -name '*.pyo' -o -name '*.exe' ')' -exec rm -f '{}' + \
 && rm -rf /tmp/* /pyenv-version.txt /python-versions.txt

COPY requirements-setup.txt /
RUN pip install -r /requirements-setup.txt \
    && find $PYENV_ROOT/versions -type d '(' -name '__pycache__' -o -name 'test' -o -name 'tests' ')' -exec rm -rf '{}' + \
    && find $PYENV_ROOT/versions -type f '(' -name '*.pyo' -o -name '*.exe' ')' -exec rm -f '{}' + \
 && rm -rf /tmp/* /requirements-setup.txt
