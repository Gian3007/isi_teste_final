FROM python:3.7-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3" ]

CMD [ "/src/app.py" ] #file to run of django








