FROM node:22-slim AS dashboard
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Wolfi provides glibc-compatible Python wheels and independently patched OS packages.
FROM cgr.dev/chainguard/wolfi-base:latest
RUN apk add --no-cache python-3.12 py3.12-pip libstdc++ libgomp bash
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . && addgroup -g 10001 runner && adduser -D -u 10001 -G runner runner
# InferScale executes Python Ray tasks only; do not ship Ray's optional Java runtime.
# Removing the actual JAR also removes its vulnerable bundled HttpComponents code.
RUN python -c "import pathlib, ray; jars = pathlib.Path(ray.__file__).parent / 'jars'; [p.unlink() for p in jars.glob('*.jar')]"
COPY configs ./configs
COPY --from=dashboard /frontend/dist ./frontend/dist
RUN mkdir /data && chown runner:runner /data
USER runner
ENV INFERSCALE_DATABASE_URL=sqlite:////data/inferscale.db
EXPOSE 8000
CMD ["inferscale", "serve", "--host", "0.0.0.0"]
