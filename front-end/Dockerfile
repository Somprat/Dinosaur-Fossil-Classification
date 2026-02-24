FROM python:3.11

WORKDIR /code

# COPY ./front-end/requirements.txt /code/requirements.txt

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY . /code/

COPY . .

EXPOSE 8000

CMD ["python", "UI.py", "--host", "0.0.0.0", "--port", "8000"]