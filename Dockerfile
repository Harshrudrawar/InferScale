FROM node:22-slim AS dashboard
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . && useradd --uid 10001 --create-home runner
COPY configs ./configs
COPY --from=dashboard /frontend/dist ./frontend/dist
RUN mkdir /data && chown runner:runner /data
USER runner
ENV INFERSCALE_DATABASE_URL=sqlite:////data/inferscale.db
EXPOSE 8000
CMD ["inferscale", "serve", "--host", "0.0.0.0"]
