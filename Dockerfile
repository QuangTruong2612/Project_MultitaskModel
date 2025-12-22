FROM python:3.10-slim
RUN apt update -y && apt install awscli -y
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. SAU ĐÓ mới copy toàn bộ code và MODEL vào
COPY . /app

RUN  pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]