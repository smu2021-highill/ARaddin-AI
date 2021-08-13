FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update

RUN apt-get install build-essential cmake -y

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "stock.wsgi:application"]
