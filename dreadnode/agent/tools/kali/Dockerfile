FROM kalilinux/kali-rolling

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install the kali metapackage
RUN apt update && apt -y install kali-linux-headless

# Install necessary packages
RUN apt-get update && apt-get install -y \
    openvpn \
    bsdextrautils \
    coreutils \
    expect \
    ftp  \
    gdb \
    gobuster \
    iputils-ping \
    jq \
    ltrace \
    medusa \
    ncat \
    poppler-utils \
    redis-tools \
    seclists \
    sshpass \
    strace \
    sublist3r \
    && rm -rf /var/lib/apt/lists/*

# Unzip rockyou.txt
RUN gunzip -f /usr/share/wordlists/rockyou.txt.gz

# Create directory for VPN configs. It will be mounted from the host in runtime
RUN mkdir -p /tmp/vpn

# Creating a file to help LLM understand this is the Docker container not the target machine :)
RUN touch /root/THIS_IS_DOCKER_CONTAINER_NOT_TARGET_MACHINE.txt

WORKDIR /root/

# Start VPN and keep container running
CMD openvpn --config /tmp/vpn/config.ovpn & \
    sleep 5 && \
    tail -f /dev/null 