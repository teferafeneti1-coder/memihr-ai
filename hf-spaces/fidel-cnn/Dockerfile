FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Hugging Face Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Run with port override via env var (app.py reads FLASK_PORT)
CMD ["python", "app.py"]
