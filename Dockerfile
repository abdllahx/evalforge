# Slim image used by the scheduler service in docker-compose.yml.
# Builds the evalforge package + claude CLI dependency.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# claude CLI for the chokepoint subprocess. Container needs ANTHROPIC_API_KEY
# at runtime — local Max OAuth does NOT carry over.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://deno.land/install.sh | sh \
 || true

COPY pyproject.toml ./
RUN pip install -U pip uv && uv pip install --system --no-cache "."

COPY src ./src
COPY scripts ./scripts
COPY sql ./sql
COPY dashboard ./dashboard

CMD ["python", "scripts/scheduler.py"]
