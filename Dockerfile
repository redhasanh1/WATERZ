FROM mcr.microsoft.com/devcontainers/javascript-node:20
WORKDIR /workspaces/watermarkz
COPY . .
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && npm install --legacy-peer-deps
CMD ["/bin/bash"]
