### **Spec: The `EngineResult` Pattern & Facade (Finalized)**

#### **1. Motivation**

The `EngineResult` pattern is introduced to finalize the separation between **Dialogue History** (the "Ledger") and **Execution Telemetry** (the "Trace").

By wrapping the state in a formal result container at the orchestration layer, we avoid the "God Object" trap in the `Conversation` model. This allows the Engine to return a complete "Proof of Work" receipt—containing tokens, durations, and cache status—without bloating the persistent message store or requiring "magic" attributes within the data layer.

---

#### **2. Structural Definition & Placement**

`EngineResult` lives in **`conduit.core.engine.result`** because it is an orchestration artifact of the FSM, not a fundamental unit of the domain data model.

* **Attributes (Data Layer):**
* `conversation`: The updated `Conversation` object (from `domain`).
* `metadata`: The `EngineMetadata` union (e.g., `ResponseMetadata`, `ToolMetadata`).


* **Properties (Ergonomic Layer):** These computed proxies maintain the "Cascade" UI pattern, allowing the result to behave like the conversation it contains.
* `.last`: Proxies to `self.conversation.last`.
* `.content`: Proxies to `self.conversation.content`.
* `.messages`: Proxies to `self.conversation.messages`.



---

#### **3. The Facade Pattern**

To maintain ergonomic simplicity for scripts and REPL usage, `EngineResult` is aliased as `ConduitResult` at the framework's entry points.

```python
# src/conduit/sync.py & src/conduit/async_.py
from conduit.core.engine.result import EngineResult as ConduitResult

# Allied ergonomic alias
from conduit.core.model.model_sync import ModelSync as Model

```

---

#### **4. Architectural Hierarchy**

The system adheres to a strict one-way dependency flow: **Core → Domain**.

1. **`domain.result.response`**: Stateless LLM output (`GenerationResponse`). Used by `Model`.
2. **`core.engine.result`**: Stateful transition result (`EngineResult`). Used by `Engine` and `Conduit`.
3. **`domain.conversation.conversation`**: The pure data container.

---

#### **5. Implementation in `ConduitBatchAsync**`

The `BatchReporter` consumes the `metadata` attribute from the `ConduitResult` to populate the live display. This ensures the batch runner has explicit visibility into the trace without the `Engine` needing to know about the UI.

```python
# ConduitBatch usage logic
result = await conduit.run(vars) # Returns a ConduitResult (Facade for EngineResult)
reporter.report_completion(
    source_id=vars["source_id"],
    is_cache=result.metadata.cache_hit,
    duration=result.metadata.duration,
    tokens=result.metadata.output_tokens
)

```

---

#### **6. Future-Proofing: Multi-Agent Handoffs**

`EngineResult` serves as the standardized **Handoff Packet**. When agents communicate, they don't just pass text; they pass the packet. The recipient uses the **Ergonomic Layer** to process the message and the **Data Layer** to audit the sender’s operational metadata.
