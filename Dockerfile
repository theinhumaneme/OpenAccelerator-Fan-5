FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/     ./src/
COPY configs/ ./configs/
COPY workload_data/ ./workload_data/

# Results written here; mount a host volume so data persists after the container exits.
VOLUME ["/app/results"]

ENTRYPOINT ["python", "src/benchmark.py"]
CMD ["--help"]
