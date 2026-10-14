FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements_runtime.txt ./
RUN python -m pip install --no-cache-dir -r requirements_runtime.txt
COPY scripts ./scripts
COPY README.md ./README.md
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--app-dir", "scripts", "--host", "0.0.0.0", "--port", "8000"]
