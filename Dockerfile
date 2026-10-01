FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu
COPY golias_dashboard.py .
COPY goliasv7_model.py .
COPY models/ models/ 2>$null || true
EXPOSE 8080
CMD ["python", "golias_dashboard.py"]
