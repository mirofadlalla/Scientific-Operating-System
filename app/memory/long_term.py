import json
import os
import logging
from threading import Lock
from typing import List, Dict, Any
import redis

logger = logging.getLogger(__name__)


class LongTermMemory:
	"""Persistent long-term memory backed by Redis with a JSON-file fallback.

	This is a lightweight store supporting adding entries and a basic
	substring/keyword search. It tries connecting to Redis, falling back
	gracefully to a JSON file if Redis is offline.
	"""

	def __init__(self, path: str | None = None, host: str = "localhost", port: int = 6379, db: int = 0):
		self.path = path or os.path.join(os.path.dirname(__file__), "long_term_store.json")
		self._lock = Lock()
		self.is_redis = False
		self.redis_client = None

		try:
			# Establish Redis connection with timeout to avoid hanging if down
			self.redis_client = redis.Redis(
				host=host,
				port=port,
				db=db,
				decode_responses=True,
				socket_connect_timeout=2.0
			)
			self.redis_client.ping()
			self.is_redis = True
			logger.info(f"✅ Connected to Redis for LongTermMemory on {host}:{port}")
		except Exception as e:
			logger.warning(f"⚠️ Redis connection failed: {e}. Falling back to JSON-backed storage.")
			self.is_redis = False

		# Initialize JSON file fallback structure
		if not self.is_redis:
			self._load_json_store()

	def _load_json_store(self) -> None:
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
		
		if self.is_redis:
			try:
				self.redis_client.rpush("long_term_memory", json.dumps(entry, ensure_ascii=False))
				return
			except Exception as e:
				logger.error(f"Redis add_entry failed: {e}. Falling back to JSON file.")
				self.is_redis = False  # Set to False so we fall back to file write

		with self._lock:
			if not hasattr(self, "_store"):
				self._load_json_store()
			self._store.append(entry)
			with open(self.path, "w", encoding="utf-8") as f:
				json.dump(self._store, f, ensure_ascii=False, indent=2)

	def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		q = (query or "").lower()
		results: List[tuple[int, Dict[str, Any]]] = []
		
		entries = None
		if self.is_redis:
			try:
				raw_entries = self.redis_client.lrange("long_term_memory", 0, -1)
				entries = [json.loads(e) for e in raw_entries]
			except Exception as e:
				logger.error(f"Redis search failed: {e}. Falling back to JSON file.")
				self.is_redis = False
				entries = None

		if entries is None:
			with self._lock:
				if not hasattr(self, "_store"):
					self._load_json_store()
				entries = list(self._store)

		for e in entries:
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
