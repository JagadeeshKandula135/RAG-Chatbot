"""Thread-safe in-memory session store with TTL eviction.

Each browser session gets isolated document text, sections, vector collection
and conversation memory, so many users can be served concurrently.
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.create_vector_db import delete_vector_db


@dataclass
class Session:
    id: str
    full_text: str = ""
    filename: str = ""
    topics: Dict[str, str] = field(default_factory=dict)
    vector_db: Any = None
    chain: Any = None
    last_access: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        self.evict_expired()
        s = Session(id=uuid.uuid4().hex)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id: Optional[str]) -> Optional[Session]:
        if not session_id:
            return None
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.last_access = time.time()
            return s

    def delete(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s:
            delete_vector_db(s.id)
        return s is not None

    def evict_expired(self) -> int:
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [i for i, s in self._sessions.items() if s.last_access < cutoff]
        for i in stale:
            self.delete(i)
        return len(stale)
