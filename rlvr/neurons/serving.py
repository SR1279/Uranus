"""Serve on a port that a newer copy of the miner can share, so the miner is replaced without a gap.

Every copy opens its port with SO_REUSEPORT, so a new copy can listen beside the running
one and the kernel shares new connections between them. When the old copy is asked to stop
(SIGINT or SIGTERM), it closes its listening socket at once, so new connections all go to
the new copy, then finishes the requests it is serving, such as a solve, and only then
exits. With net.ipv4.tcp_migrate_req set to 1, connections still waiting in the old copy's
queue are handed to the new copy instead of being reset.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import uvicorn

BACKLOG = 2048
# How long a stopping copy keeps open connections that have not sent a request yet.
SETTLE_S = 1.0


def shared_socket(host: str, port: int) -> socket.socket:
    """A listening socket that other copies may open on the same address at the same time."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, port))
        sock.listen(BACKLOG)
    except OSError:
        sock.close()
        raise
    return sock


class DrainingServer(uvicorn.Server):
    """uvicorn's server, except that it stops taking connections a moment before it shuts the open ones.

    uvicorn's shutdown closes a connection that has not sent its request yet as if it were idle,
    which resets a client the kernel handed over just before the stop. Closing the listening
    socket first and pausing lets such a connection send its request, which is then answered.
    """

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        for server in self.servers:
            server.close()
        for sock in sockets or []:
            sock.close()
        await asyncio.sleep(SETTLE_S)
        await super().shutdown(sockets=sockets)


def serve(app: Any, host: str, port: int, *, drain_s: float, log_level: str = "info") -> None:
    """Serve `app` until a stop signal, then finish open requests for up to drain_s seconds."""
    server = DrainingServer(uvicorn.Config(app, log_level=log_level, timeout_graceful_shutdown=int(drain_s)))
    try:
        server.run(sockets=[shared_socket(host, port)])
    except KeyboardInterrupt:
        # asyncio's runner re-raises the stop signal once the server has drained and shut down;
        # the shutdown is complete by then, so the retiring copy exits quietly instead of with a traceback.
        return
