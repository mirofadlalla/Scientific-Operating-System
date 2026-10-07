"""Prompt fragments shared by every answer-generating prompt.

Dependency-free on purpose so any module (agents, orchestration, RAG) can import it
without pulling in the rest of the app.
"""

CONCISE_ANSWER_RULE = (
    "ANSWER LENGTH: Keep every answer short and to the point - at most 3-4 sentences "
    "(roughly 60 words) unless the user explicitly asks for more detail. "
    "Start with the direct answer. Skip introductions, restating the question, "
    "disclaimers and long lists; keep only the key facts and numbers the question needs. "
    "Your reply is often spoken aloud, so prefer plain short sentences over tables or bullet lists."
)
