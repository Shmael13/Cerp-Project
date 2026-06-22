FROM python:3.11-slim

WORKDIR /app

# Install system deps for wget
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Bundle Plotly.js locally so the app has no CDN dependency
RUN mkdir -p static/js && \
    wget -q -O static/js/plotly.min.js https://cdn.plot.ly/plotly-2.35.2.min.js

# Hugging Face Spaces runs on port 7860
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
