from collections import deque, defaultdict
from threading import Lock
from typing import List, Dict, Any


class ShortTermMemory:
	"""Lightweight in-memory short-term memory per session.

	This is a thread-safe ring buffer for recent messages. It is
	intentionally simple so it works without external dependencies.
	"""

	def __init__(self, maxlen: int = 50):
		self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
		self._lock = Lock()

	def add_message(self, session_id: str, role: str, content: str) -> None:
		with self._lock:
			self._store[session_id].append({"role": role, "content": content})

	def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
		with self._lock:
			return list(self._store.get(session_id, []))[-limit:]

	def clear(self, session_id: str) -> None:
		with self._lock:
			if session_id in self._store:
				self._store[session_id].clear()

