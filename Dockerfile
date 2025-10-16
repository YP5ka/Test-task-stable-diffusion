FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

# Default command can be overridden (e.g., with `-- generate ...`)
CMD ["python", "src/lora_fit_and_save.py"]


