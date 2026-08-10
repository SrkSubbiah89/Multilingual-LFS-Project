"""
eval/legacy_decision_policy41/

Task 41: Historical Decision-Policy Compatibility Study.

Explicit non-literal label: this package does NOT claim to be, and must
never be described as, an exact reproduction, literal old implementation
execution, or old paper result reproduced. Tasks 39/40/40.1 closed the
strict literal-reproduction path (the contemporaneous qdrant-client
version recovered for LEGACY_SHA, 1.17.0, does not expose
QdrantClient.search()). This package instead implements and hermetically
tests, as a small additive component, the historical conditional LLM
decision policy from `backend/agents/isco_classifier.py` @ LEGACY_SHA --
independent of Qdrant, any embedding model, CrewAI, or any live LLM/
provider call -- so the policy's *decision logic* (five ordered
candidates, the 0.92 confidence threshold, exactly-one reranker
invocation below threshold, candidate-only selection, the historical
prompt field set, and the semantic/llm_ranked method labels) can be
verified against real, quoted legacy source text without executing any
part of the obsolete runtime.
"""

LEGACY_SHA = "824fcf235ae2f8787706cf479a07620519c914de"
