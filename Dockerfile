FROM python:3.10.13-slim

WORKDIR /usr/src
COPY translate ./translate
COPY predict ./predict
COPY app ./app
COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY . .
RUN rm -rf dataset
RUN rm -rf django
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
