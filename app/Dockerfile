FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system build deps (needed by some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
  chromium \
  fonts-liberation \
  libxss1 \
  libappindicator3-1 \
  libatk-bridge2.0-0 \
  libgtk-3-0 \
  libnss3 \
  libasound2 \
  libgbm1 \
  xdg-utils \
  ca-certificates \
  && ln -s /usr/bin/chromium /usr/bin/google-chrome || true \
  && rm -rf /var/lib/apt/lists/*

ENV CHROME_BIN=/usr/bin/chromium

# Copy and install Python dependencies first to leverage docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r /app/requirements.txt

# Copy app source
COPY . /app

# Streamlit default port
EXPOSE 8501

# Run the app (bind to all interfaces so container is reachable externally)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
