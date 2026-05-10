#!/usr/bin/env bash
# Open a new terminal (shell) in the same running Polaris container.
# Use after run.sh: exit that shell, then run ./terminal.sh to get another shell in the same container.

CONTAINER_NAME="${CONTAINER_NAME:-polaris-dev}"
DISPLAY="${DISPLAY:-:0}"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Error: Container ${CONTAINER_NAME} is not running. Start it with ./run.sh first." >&2
  exit 1
fi

exec docker exec -it -e DISPLAY="${DISPLAY}" "${CONTAINER_NAME}" /bin/bash
