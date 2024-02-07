FROM python:3.10.13-slim

WORKDIR /usr/src/app

RUN pip install virtualenv
RUN python -m venv python
RUN . python/bin/activate

COPY ./requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .

WORKDIR /usr/src/app/api
CMD ["uvicorn", "detect_suicide_api:app","--host", "0.0.0.0", "--port", "80"]

EXPOSE 80