FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run unprivileged. data/ (history, saved runs, embedding model) is the only writable path.
RUN useradd --create-home appuser && mkdir -p data && chown -R appuser:appuser data
USER appuser

EXPOSE 8501

# Railway and Render provide $PORT; fall back to Streamlit's default locally. exec makes Streamlit
# PID 1 so it receives the platform's stop signal and shuts down cleanly.
CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
