FROM python:3.11.2

RUN apt-get update && apt-get install -y \
    git 

WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python3", "app.py"]
