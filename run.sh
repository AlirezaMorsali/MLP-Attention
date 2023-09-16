#!/usr/bin/env sh
set -e

docker compose -f docker-compose.yaml build dev
docker compose -f docker-compose.yaml run dev
