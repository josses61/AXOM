FROM node:22-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    g++ \
    gcc \
    golang-go \
    python3 \
    rustc \
  && rm -rf /var/lib/apt/lists/*

COPY runner.js /opt/axon-runner.js
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["node", "/opt/axon-runner.js"]
