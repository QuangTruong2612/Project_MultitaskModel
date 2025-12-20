FROM python:3.10-slim
RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]