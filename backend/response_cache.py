"""First-turn response cache (plan #2).

A small thread-safe TTL + LRU map from a normalized question to its finished
answer. Deliberately exact-match (no semantic similarity) and first-turn only —
follow-up answers depend on session history, and semantic "close enough"
matching risks serving a wrong answer. In-process by design: a restart (every
deploy / blue-green corpus cutover) clears it, so a cached answer can never
outlive the corpus it was generated against.
"""

import threading
import time
from collections import OrderedDict
from typing import Optional


def normalize_question(question: str) -> str:
    """Cache key: lowercased, whitespace-collapsed. Exact match only — two
    questions share an answer iff they normalize identically."""
    return " ".join(question.strip().lower().split())


class ResponseCache:
    def __init__(self, maxsize: int = 1000, ttl_s: int = 86400):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._store: "OrderedDict[str, tuple]" = OrderedDict()  # key -> (expires_at, answer)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            expires_at, answer = item
            if expires_at < now:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)  # LRU touch
            self._hits += 1
            return answer

    def set(self, key: str, answer: str) -> None:
        with self._lock:
            self._store[key] = (time.time() + self.ttl_s, answer)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)  # evict oldest

    def stats(self) -> dict:
        with self._lock:
            lookups = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "hit_rate": round(self._hits / lookups, 3) if lookups else None,
            }
