FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_MODE=dashboard \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.lock .
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && useradd --create-home --uid 10001 appuser \
    && chown appuser:appuser /app
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser examples ./examples
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser app.py api.py ui.py sensor.py combined.py ./
USER appuser
# Explicit build-time training; no training on startup or requests.
RUN python app.py train \
    && python sensor.py download \
    && python sensor.py train \
    && python combined.py train
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "src.hosting", "healthcheck"]
CMD ["python", "-m", "src.hosting"]
