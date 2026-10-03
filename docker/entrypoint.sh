#!/bin/sh
# Container entrypoint.
#
# The Chemical agent talks to a local MCP server (http://localhost:8001/mcp) that
# nothing else in the project starts, so we launch it here next to the API.
# Set START_MCP_SERVER=0 if you run the MCP server as a separate service and
# point CHEMICAL_MCP_URL at it instead.
set -eu

if [ "${START_MCP_SERVER:-1}" = "1" ]; then
  # Tiny supervisor loop: restart the MCP server if it ever crashes.
  # It dies with the container (tini -g forwards SIGTERM to the whole group).
  (
    while true; do
      python -m app.agents.chemical.mcp.chemical_server || true
      echo "[entrypoint] MCP server exited — restarting in 2s" >&2
      sleep 2
    done
  ) &
fi

exec "$@"
