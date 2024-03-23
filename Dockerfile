# Stage 1 : build
FROM python:3.10.11-slim

WORKDIR /app

COPY requirements.txt .

# RUN pip install -r requirements.txt
RUN  pip install -r requirements.txt

# Copy app code
COPY . . 

EXPOSE 5000

CMD [ "./run_server.sh" ]