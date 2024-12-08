FROM python:3.9

RUN apt-get update && \
    apt-get install -y \
        zip \
        python3-pip \
        python3-dev

RUN pip3 install psutil scipy nibabel pyradiomics SimpleITK numpy==1.26.4

RUN useradd -ms /bin/bash python
RUN echo "python:password" | chpasswd
RUN adduser python sudo
USER python
WORKDIR /home/python

COPY data.zip data.zip
COPY client.py client.py
COPY client.sh client.sh

COPY DataHandler.py DataHandler.py
COPY DataPoint.py DataPoint.py
COPY extractor_params.py extractor_params.py
COPY LayeredArray.py LayeredArray.py
COPY util.py util.py
COPY visual.py visual.py

RUN mkdir -p mount

ENV MINIMAL=1

CMD '/home/python/client.sh'
