# Need docker >= 20.10.9, see https://stackoverflow.com/questions/71941032/why-i-cannot-run-apt-update-inside-a-fresh-ubuntu22-04

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
WORKDIR /app

# COPY envpool/docker/certs/AselsanSubCA.crt /usr/local/share/ca-certificates/
# COPY envpool/docker/certs/AselsanRootCA.crt /usr/local/share/ca-certificates/
# COPY docker/certs/AselsanRootCA.crt /usr/local/share/ca-certificates/
# ARG PATH=$PATH:$HOME/go/bin
RUN apt update && apt upgrade -y && \
    apt install -y build-essential cmake git sudo \
    npm wget curl zsh tmux vim golang-1.21 clang-format clang-tidy swig qtdeclarative5-dev  \
    python3-pip libgoogle-glog-dev libeigen3-dev libboost-all-dev libgflags2.2 libgflags-dev \
    libglfw3-dev libopengl-dev libtinyxml2-dev libglew-dev libglib2.0-dev libcurses-ocaml-dev python3-dev \
    libxinerama-dev libxcursor-dev libxi-dev cmake g++ ca-certificates liblcm-dev 

RUN npm install -g @bazel/bazelisk

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

RUN pip3 install gym numpy matplotlib  jax jaxlib flax optax   
RUN export USE_BAZEL_VERSION=6.5.0
WORKDIR /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

