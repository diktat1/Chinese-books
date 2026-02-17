FROM python:3.11-slim

# System deps for lxml, audio, and optional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user "user" (uid 1000)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app code
COPY . .

# Own everything by the runtime user
RUN chown -R user:user /home/user/app

USER user

EXPOSE 7860

CMD ["gunicorn", "web_app:app", "--bind", "0.0.0.0:7860", "--timeout", "600", "--workers", "1", "--threads", "8"]
