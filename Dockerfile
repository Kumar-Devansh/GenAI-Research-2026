FROM python:3.11-slim
WORKDIR /app
COPY requirements_part2.txt .
RUN pip install --no-cache-dir -r requirements_part2.txt
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--app-dir", "scripts", "--host", "0.0.0.0", "--port", "8000"]
