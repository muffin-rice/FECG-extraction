FROM python:3.10-alpine
RUN mkdir /app
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "code/vae_train.py"]