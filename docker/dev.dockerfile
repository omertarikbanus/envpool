# Need docker >= 20.10.9, see https://stackoverflow.com/questions/71941032/why-i-cannot-run-apt-update-inside-a-fresh-ubuntu22-04

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
WORKDIR /app

# COPY envpool/docker/certs/AselsanSubCA.crt /usr/local/share/ca-certificates/
# COPY envpool/docker/certs/AselsanRootCA.crt /usr/local/share/ca-certificates/
# COPY docker/certs/AselsanRootCA.crt /usr/local/share/ca-certificates/
# ARG PATH=$PATH:$HOME/go/bin
RUN apt update 
RUN apt update && apt upgrade -y && \
    apt install -y \
    # Build essentials
    build-essential \
    ca-certificates \
    cmake \
    htop \
    g++ \
    git \
    sudo \
    # Development tools
    clang-format \
    clang-tidy \
    curl \
    tmux \
    vim \
    wget \
    zsh \
    # Programming languages and tools
    golang-1.21 \
    npm \
    python3-dev \
    python3-pip \
    swig \
    # Graphics and UI libraries
    kst \
    libegl1 \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libopengl-dev \
    libosmesa6 \
    libosmesa6-dev \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb-util1 \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    qtdeclarative5-dev \
    xvfb \
    ffmpeg \
    # Other libraries
    libboost-all-dev \
    libcurses-ocaml-dev \
    libeigen3-dev \
    libgflags-dev \
    libgflags2.2 \
    libglib2.0-dev \
    libgoogle-glog-dev \
    liblcm-dev \
    libtinyxml2-dev \
    libx11-xcb-dev \
    libx11-xcb1 \
    libxcb1

RUN pip3 install mujoco gym numpy matplotlib  jax==0.5.3  flax optax stable-baselines3 tensorboard "shimmy>=2.0"
# XLA Extensions
# RUN pip3 install mujoco gym matplotlib  jax==0.4.13 jaxlib==0.4.13 "numpy<2.0"  flax optax stable-baselines3  tensorboard "shimmy>=2.0"  optuna>=3.0.0  plotly>=5.0.0  


RUN npm install -g @bazel/bazelisk
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"


RUN git clone https://github.com/gpakosz/.tmux.git

RUN ln -s -f .tmux/.tmux.conf
RUN cp .tmux/.tmux.conf.local .
RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local
RUN echo "export PATH=$PATH:$HOME/go/bin" >> .zshrc

RUN useradd -ms /bin/zsh github-action

WORKDIR /app
#     RUN go install github.com/bazelbuild/bazelisk@latest 

# RUN ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
# RUN go install github.com/bazelbuild/buildtools/buildifier@latest
# RUN $HOME/go/bin/bazel version
# RUN ln -s /usr/bin/python3 /usr/bin/python
# RUN ln -sf /usr/lib/go-1.21/bin/go /usr/bin/go
# RUN export PATH=$HOME/go/bin:$PATH

# RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
WORKDIR $HOME
# RUN git clone https://github.com/gpakosz/.tmux.git
# RUN ln -s -f .tmux/.tmux.conf
# RUN cp .tmux/.tmux.conf.local .
# RUN echo "set-option -g default-shell /bin/zsh" >> .tmux.conf.local
# RUN echo "set-option -g history-limit 10000" >> .tmux.conf.local
# RUN echo "export PATH=$PATH:$HOME/go/bin" >> .zshrc


RUN git clone --branch 2.3.7 https://github.com/deepmind/mujoco.git /root/mujoco && \
    cd /root/mujoco && \
    mkdir -p build && \
    cd build && \
    cmake -DMJ_OSMESA=ON .. && \
    cmake --build . --parallel  && \
    cmake --install .
RUN ldconfig
RUN echo "export USE_BAZEL_VERSION=6.5.0" >> /root/.bashrc
RUN echo "export USE_BAZEL_VERSION=6.5.0" >> /root/.zshrc

WORKDIR /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN ln -s /usr/include/aarch64-linux-gnu/qt5 /usr/include/qt
RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
ENTRYPOINT ["/entrypoint.sh"]

