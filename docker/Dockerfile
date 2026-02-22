FROM python:3.12-slim

# Non-root user
RUN groupadd -r llmbench && useradd -r -g llmbench -d /app -s /sbin/nologin llmbench

WORKDIR /app

# Git is needed for benchmark trigger (commit + push to inbox)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r /app/web/requirements.txt

# Copy application code
COPY web/ /app/web/
COPY config/ /app/config/
COPY core/__init__.py /app/core/__init__.py

# Ensure result directories exist
RUN mkdir -p /app/results/human /app/results/questions /app/results/history \
    /app/.claude/inbox

# Ownership
RUN chown -R llmbench:llmbench /app

USER llmbench

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/datasets')" || exit 1

CMD ["gunicorn", "web.server:app", \
     "-b", "0.0.0.0:8080", \
     "-w", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
