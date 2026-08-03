# Requires `docker build -t voptrade:latest .` (root Dockerfile) to be built first.
ARG sourceimage=voptrade
ARG sourcetag=latest
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-plot.txt /voptrade/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
