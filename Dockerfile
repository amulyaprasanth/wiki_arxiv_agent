FROM python:3.10-slim

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["streamlit", "run", "wisdom_retriever.py", "--server.port", "8001",  "--server.address=0.0.0.0"]