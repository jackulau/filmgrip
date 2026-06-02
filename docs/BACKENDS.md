# Planner backends

film-grip's planner is **provider-agnostic**. The validate→apply pipeline (and the CLI, the panel,
the repair loop) only ever talk to a small seam — they never reference a specific LLM. Swapping or
adding a provider is therefore a self-contained job: implement one transport, register one name.

## The seam

Two layers, both in `filmgrip/integration/`:

| Layer | Type | Job |
|---|---|---|
| `Transport` (`mcp_host.py`) | run one planning turn | `run(...) -> PlanResponse` |
| `PlannerBackend` (`backend.py`) | name a provider | `name: str`, `transport() -> Transport` |

A **backend** is selected by name (`--backend`, then `$FILMGRIP_BACKEND`, then the default
`claude`) and hands its **transport** to `plan_with_repair`. That's the whole contract.

```python
class PlannerBackend(Protocol):
    name: str
    def transport(self) -> Transport: ...
```

Register a backend so the CLI can find it:

```python
from filmgrip.integration.backend import register
register("mybackend", MyBackend)   # MyBackend() -> has .name and .transport()
```

## The transport contract

```python
def run(self, *, system_prompt: str, user_prompt: str, schema: dict,
        ctx: PlannerContext, session_id: str | None = None,
        model: str | None = None) -> PlanResponse: ...
```

A `run` MUST:

1. **Produce an EditPlan that conforms to `schema`** (a JSON Schema derived from
   `filmgrip.protocol.editplan`). Put it on `PlanResponse.structured_output` and set
   `subtype = "success"`. film-grip parses + validates it host-side against the live IR — a
   hallucinated clip id or out-of-bounds frame fails validation, never the editor.
2. **Give the model the context it needs.** Claude *pulls* FGX context via the in-process MCP tools
   (`get_selection` / `get_context` / `query_clips`). A model without MCP tool-use can instead be
   *pushed* the FGX bundle inline — `filmgrip.serialize.fgx.bundle(ctx.ir, ctx.selection.ids)` is
   the same token-frugal projection; embed it in the prompt.
3. **Report failure honestly, never crash.** Return `PlanResponse(subtype="error", errors=[...])`
   for a hard failure, or `subtype="error_max_structured_output_retries"` when the model can't
   conform after its own retries (film-grip treats that as terminal — no repair loop). `plan_edit`
   also wraps `run` in a guard, so even a raised exception becomes a clean "planner backend failed"
   result rather than a traceback.
4. **Carry `session_id`** through if the provider supports resuming a conversation, so film-grip's
   repair loop can feed validation errors back into the same context. If it can't resume, run your
   own repair inside `run` and return the final plan.

### Structured output for non-Claude models (the Codex/GPT case)

Claude has a native `output_format = {"type": "json_schema", ...}` mode; most others don't. A
Codex/GPT backend must coerce conformance itself. Recommended order:

1. Use the provider's strict JSON / function-calling / response-format feature with the EditPlan
   schema if it has one.
2. Otherwise: put the schema in the system prompt, demand "JSON only", parse the reply, and on a
   parse/validation miss run a **bounded local repair** (re-prompt with the exact error) before
   returning — mirroring what `plan_with_repair` does for Claude. Cap the retries.
3. Set `cost_usd` / `usage` if the provider reports them, so `film-grip`'s cost line stays honest.

## Status

| Backend | Name | State |
|---|---|---|
| Claude (Agent SDK + in-process MCP) | `claude` | **live** (flagship, default) |
| Codex / GPT | `codex` | **seam ready, not implemented** — resolves and fails honestly |

`film-grip edit --backend codex …` resolves the backend and returns a clear "not yet implemented"
message (it does not crash or silently fall back). Implementing it = filling in
`CodexTransport.run` in `filmgrip/integration/backend_codex.py` per the contract above.
