import json
import os
from threading import Lock
from typing import List, Dict, Any


class LongTermMemory:
	"""Simple persistent long-term memory implemented as a JSON-backed store.

	This is a lightweight fallback for real vector DBs. It supports adding
	entries and a basic substring/keyword search that returns the top matches.
	"""

	def __init__(self, path: str | None = None):
		self.path = path or os.path.join(os.path.dirname(__file__), "long_term_store.json")
		self._lock = Lock()
		if not os.path.exists(self.path):
			with open(self.path, "w", encoding="utf-8") as f:
				json.dump([], f)

		with open(self.path, "r", encoding="utf-8") as f:
			try:
				self._store: List[Dict[str, Any]] = json.load(f)
			except Exception:
				self._store = []

	def add_entry(self, session_id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
		entry = {"session_id": session_id, "text": text, "metadata": metadata or {}}
		with self._lock:
			self._store.append(entry)
			with open(self.path, "w", encoding="utf-8") as f:
				json.dump(self._store, f, ensure_ascii=False, indent=2)

	def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		q = (query or "").lower()
		results: List[tuple[int, Dict[str, Any]]] = []
		with self._lock:
			for e in self._store:
				text = (e.get("text") or "").lower()
				if not text:
					continue
				score = 0
				if q in text:
					score += 5
				# reward word overlaps
				for w in q.split():
					if w and w in text:
						score += 1
				if score > 0:
					results.append((score, e))

		results.sort(key=lambda x: x[0], reverse=True)
		return [e for _, e in results[:top_k]]

