# /app/healthcheck.py
import os, sys
try:
    import redis
except Exception:
    sys.exit(1)

u = os.getenv("REDIS_URL")
if not u:
    sys.exit(1)
try:
    r = redis.from_url(u, decode_responses=True)
    ok = r.ping()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
