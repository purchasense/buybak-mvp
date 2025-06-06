FROM nikolaik/python-nodejs:python3.11-nodejs16-slim

WORKDIR /app
RUN mkdir -p /app/TradesCSV
COPY helper.py /app
COPY TradesCSV /app/TradesCSV/

COPY . .

RUN pip install -r requirements.txt && pip cache purge && \
    cd react_frontend && npm install && npm install -g serve && cd .. 

# Flask
CMD ["sh", "launch_app.sh"]
EXPOSE 5601 3000
