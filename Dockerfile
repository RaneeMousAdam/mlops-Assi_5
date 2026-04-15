FROM python:3.10-slim

ARG RUN_ID

RUN pip install mlflow scikit-learn

RUN echo "Downloading model for Run ID: ${RUN_ID}"

WORKDIR /app
COPY train.py .
COPY check_threshold.py .

CMD ["echo", "Model server ready"]
