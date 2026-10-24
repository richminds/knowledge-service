"""Strategy registry and the unified router (I-01, I-03, I-05).

RAG_MODE / request rag_mode:
  semantic | hybrid | hybrid_graph | graph | agentic
                               - use that strategy
  auto                         - route by question complexity (from query understanding):
                                 simple -> hybrid, relationship/global -> graph,
                                 multi_step -> agentic (RAG_ROUTER_MAP), falling back to the
                                 closest enabled mode
Escalation (I-03): when an answer comes back low-confidence, the service may retry with the
next more capable enabled mode (semantic -> hybrid -> hybrid_graph -> agentic; graph ->
hybrid_graph -> agentic).
"""

from app.core.config import RagMode
from app.core.exceptions import RagModeError
from app.rag.strategies.base import RagStrategy

DEFAULT_ROUTES = {
    "simple": RagMode.HYBRID,
    "relationship": RagMode.GRAPH,
    "global": RagMode.GRAPH,
    "multi_step": RagMode.AGENTIC,
}
# Preference order when the routed mode is not enabled.
_S, _H, _HG, _G, _A = (
    RagMode.SEMANTIC,
    RagMode.HYBRID,
    RagMode.HYBRID_GRAPH,
    RagMode.GRAPH,
    RagMode.AGENTIC,
)
_FALLBACKS = {
    _S: [_S, _H, _HG, _A, _G],
    _H: [_H, _S, _HG, _A, _G],
    _HG: [_HG, _G, _H, _A, _S],
    _G: [_G, _HG, _A, _H, _S],
    _A: [_A, _HG, _G, _H, _S],
}
# I-03: the more capable modes to retry with, in order (graph-only is never a step up from
# hybrid_graph, which already includes it).
_ESCALATIONS = {
    _S: [_H, _HG, _G, _A],
    _H: [_HG, _G, _A],
    _G: [_HG, _A],
    _HG: [_A],
    _A: [],
}


class RagStrategyRegistry:
    def __init__(
        self,
        strategies: dict[RagMode, RagStrategy],
        default_mode: RagMode,
        allow_override: bool = True,
        routes: dict[str, RagMode] | None = None,
    ) -> None:
        self._strategies = {m: s for m, s in strategies.items() if m != RagMode.AUTO}
        self._default = default_mode
        self._allow_override = allow_override
        self._routes = {**DEFAULT_ROUTES, **(routes or {})}

    @property
    def enabled_modes(self) -> list[RagMode]:
        return list(self._strategies)

    @property
    def default_mode(self) -> RagMode:
        return self._default

    @property
    def allow_override(self) -> bool:
        return self._allow_override

    def route(self, complexity: str) -> RagMode:
        """I-01: the cheapest enabled approach suited to the question type."""
        preferred = self._routes.get(complexity, RagMode.HYBRID)
        for mode in _FALLBACKS.get(preferred, _FALLBACKS[_H]):
            if mode in self._strategies:
                return mode
        raise RagModeError("No RAG mode is enabled")

    def resolve(
        self, requested: RagMode | None = None, complexity: str = "simple", force: bool = False
    ) -> RagStrategy:
        """`force` honours `requested` even when overrides are disabled (admin validation)."""
        mode = self._default
        if requested is not None and requested != self._default:
            if not (self._allow_override or force):
                raise RagModeError("Per-request rag_mode is disabled (RAG_ALLOW_MODE_OVERRIDE)")
            mode = requested
        if mode == RagMode.AUTO:
            mode = self.route(complexity)
        strategy = self._strategies.get(mode)
        if strategy is None:
            enabled = ", ".join(m.value for m in self._strategies) or "none"
            raise RagModeError(f"RAG mode '{mode.value}' is not enabled (enabled: {enabled})")
        return strategy

    def escalate(self, current: RagMode) -> RagStrategy | None:
        """I-03: next more capable enabled strategy, or None."""
        for mode in _ESCALATIONS.get(current, []):
            if mode in self._strategies:
                return self._strategies[mode]
        return None
