FROM python:alpine3.7

WORKDIR /isi_test/src

COPY . /isi_test/src

RUN pip install -r /isi_test/src/requirements.txt

CMD python ./isi_test/src/app.py





