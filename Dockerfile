FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install project
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY . .

# Install the package itself
RUN uv sync --frozen --no-dev

# Create workspace directory
RUN mkdir -p /workspace && \
    groupadd --system miniagent && \
    useradd --system --gid miniagent --create-home miniagent && \
    chown -R miniagent:miniagent /app /workspace

USER miniagent

ENTRYPOINT ["uv", "run", "python", "-m", "mini_agent.cli"]
CMD ["--workspace", "/workspace"]
