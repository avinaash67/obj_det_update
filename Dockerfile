FROM nvidia/cuda:10.2-base

RUN apt-get update  &&  apt-get install -y python3.7 && apt-get install -y python3-pip libsm6 libxext6 libxrender-dev

WORKDIR /detection

COPY ./config.yaml ./main_det.py ./obj_det_functions.py ./requirements.txt ./

# ENV PYTHONUNBUFFERED=1 

RUN pip3 install -r requirements.txt

CMD ["python3","main_det.py"]