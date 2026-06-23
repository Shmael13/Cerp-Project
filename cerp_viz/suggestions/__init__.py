"""
Suggester package.

Importing this package registers all built-in suggesters:
  "Rule-Based"                    — heuristic (column types, cardinalities, name patterns)
  "Statistical"                   — statistical analysis (correlation, OLS, Pareto, …)
  "Smart (Statistical + Rule-Based)" — composite: best of both per chart type

Use suggester_registry to list available names and get instances by name.
The UI reads suggester_registry.names() and calls suggester_registry.get(name).

Adding a new suggester:
  1. Create cerp_viz/suggestions/my_suggester.py
  2. Define class MyFoo(BaseSuggester): …
  3. Call register("My Foo", MyFoo) at module level
  4. Import it here so registration fires on package load
  Nothing else changes.
"""
# Trigger self-registration (order matters: composite imports the others)
from cerp_viz.suggestions import rule_based as _rb   # noqa: F401
from cerp_viz.suggestions import statistical as _st  # noqa: F401
from cerp_viz.suggestions import composite as _cp    # noqa: F401

# Conditionally register AI suggester if anthropic + API key are present
try:
    import os
    import anthropic as _anthropic  # noqa: F401
    if os.environ.get("ANTHROPIC_API_KEY"):
        from cerp_viz.suggestions import ai_suggester as _ai  # noqa: F401
        from cerp_viz.suggestions.ai_suggester import AISuggester
        from cerp_viz.suggestions.registry import register as _reg
        _reg("AI (Claude)", AISuggester)
except ImportError:
    pass

# Conditionally register Groq suggester if groq package + API key are present
try:
    import os as _os
    from cerp_viz.ai.groq_client import is_available as _groq_ok
    if _groq_ok():
        from cerp_viz.suggestions.groq_suggester import GroqSuggester
        from cerp_viz.suggestions.registry import register as _reg2
        _reg2("AI (Groq)", GroqSuggester)
except Exception:
    pass

from cerp_viz.suggestions import registry as suggester_registry  # noqa: F401
from cerp_viz.core.suggestions import BaseSuggester              # noqa: F401

__all__ = ["suggester_registry", "BaseSuggester"]
