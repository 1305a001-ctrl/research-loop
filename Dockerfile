FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim
LABEL org.opencontainers.image.source=https://github.com/1305a001-ctrl/research-loop
LABEL org.opencontainers.image.description="Continuous research loop — Sharpe tracker + auto-halt"
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
RUN useradd --create-home --shell /bin/bash rl
USER rl
EXPOSE 8015
CMD ["python", "-m", "research_loop.main"]
