FROM python:3.11

RUN apt-get update -y

RUN pip install --upgrade pip

# Install experiment dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY . /app

WORKDIR /app
