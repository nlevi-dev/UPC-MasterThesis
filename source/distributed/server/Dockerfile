FROM python:3.9

RUN apt-get update && \
    apt-get install -y \
        zip \
        python3-pip \
        python3-dev

RUN pip3 install flask

RUN useradd -ms /bin/bash python
RUN echo "python:password" | chpasswd
RUN adduser python sudo
USER python
WORKDIR /home/python

COPY server.py server.py

RUN mkdir -p mount

CMD 'python /home/python/server.py'
