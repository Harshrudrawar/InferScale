import time
from collections import deque

from starlette.responses import JSONResponse


class RequestLimits:
    """Single-process global POST rate and body limits; edge proxy needed for multi-node use."""

    def __init__(self, app, max_bytes=1024 * 1024, per_minute=600):
        self.app = app
        self.max_bytes = max_bytes
        self.per_minute = per_minute
        self.arrivals = deque()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        now = time.monotonic()
        while self.arrivals and self.arrivals[0] <= now - 60:
            self.arrivals.popleft()
        if len(self.arrivals) >= self.per_minute:
            return await JSONResponse(
                {"detail": "Global request rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )(scope, receive, send)
        self.arrivals.append(now)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > self.max_bytes:
                return await JSONResponse(
                    {"detail": "Request body exceeds 1 MiB"}, status_code=413
                )(scope, receive, send)
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
