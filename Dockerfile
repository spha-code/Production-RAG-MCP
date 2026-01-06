FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir uv
WORKDIR /app

# bump timeout first
ENV UV_HTTP_TIMEOUT=300

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python python3.12 --extra cpu

# FIX: Copy the directories into named subdirectories
COPY backend ./backend
COPY web ./web

# Now "backend.app:app" will work because the "backend" folder exists
CMD ["uv", "run", "--python", "python3.12", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]