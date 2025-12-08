# LangSmith Integration Specification

## Purpose

This specification describes how to integrate LangSmith into the WINK ecosystem
for observability, prompt management, and evaluation capabilities. It covers
the implementation of a LangSmith-compatible prompt overrides store and
end-to-end tracing configuration.

**See also**: [LANGGRAPH_ADAPTER.md](./LANGGRAPH_ADAPTER.md) for the LangGraph
adapter specification, which integrates naturally with LangSmith.

## Guiding Principles

- **Centralized prompt management**: Enable prompt iteration via LangSmith Hub
  without source code changes.
- **Observable execution**: Automatic tracing for debugging and evaluation.
- **Version stability**: Support commit hash pinning for production deployments.
- **Non-invasive integration**: LangSmith features are opt-in and don't affect
  core WINK functionality.

## Environment Configuration

```python
import os

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "my-wink-project"

# Optional: Custom endpoint
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
```

When these environment variables are set, all LangGraph executions automatically
send traces to LangSmith without additional configuration.

## LangSmith Prompt Hub Overrides Store

The `LangSmithPromptOverridesStore` implements WINK's `PromptOverridesStore`
protocol using LangSmith's Prompt Hub as the backing store. This enables:

- **Centralized prompt management**: Edit prompts in LangSmith's Playground
- **Version control**: Each push creates a commit hash for rollback
- **Team collaboration**: Share prompts across organization members
- **A/B testing**: Use tags to manage production vs experimental variants

### Store Configuration

```python
from langsmith import Client
from weakincentives.prompt.overrides import (
    PromptOverridesStore,
    PromptOverride,
    SectionOverride,
    ToolOverride,
)
from weakincentives.prompt.rendering import PromptDescriptor

@FrozenDataclass()
class LangSmithStoreConfig:
    """Configuration for LangSmith prompt store.

    Attributes:
        api_key: LangSmith API key. Falls back to LANGSMITH_API_KEY env var.
        api_url: LangSmith API endpoint. Falls back to LANGSMITH_ENDPOINT.
        organization: Organization name for prompt namespacing.
        default_tag: Default version tag. Use "latest" for development,
            commit hashes for production stability.
    """

    api_key: str | None = None
    api_url: str | None = None
    organization: str | None = None
    default_tag: str = "latest"
```

### Store Implementation

```python
class LangSmithPromptOverridesStore(PromptOverridesStore):
    """Prompt overrides store backed by LangSmith Prompt Hub.

    This store maps WINK's prompt override model to LangSmith's prompt
    versioning system:

    - WINK namespace/key → LangSmith repo name: "{org}/{ns}--{prompt_key}"
    - WINK tag → LangSmith tag or commit hash
    - WINK section overrides → LangSmith prompt template variables
    - WINK content hashes → Stored in prompt metadata for validation

    Example LangSmith prompt structure:
        Repo: "myorg/webapp-agents--welcome"
        Template: "{system_section}\n\n{task_section}"
        Metadata: {"wink_hashes": {"system": "abc123...", "task": "def456..."}}
    """

    def __init__(self, config: LangSmithStoreConfig | None = None) -> None:
        self._config = config or LangSmithStoreConfig()
        self._client = Client(
            api_key=self._config.api_key,
            api_url=self._config.api_url,
        )

    def _repo_name(self, ns: str, prompt_key: str) -> str:
        """Build LangSmith repo name from WINK identifiers."""
        prefix = f"{self._config.organization}/" if self._config.organization else ""
        # Replace path separators with double-dash for flat namespace
        safe_ns = ns.replace("/", "--")
        return f"{prefix}{safe_ns}--{prompt_key}"

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Pull prompt override from LangSmith Hub.

        Args:
            descriptor: WINK prompt descriptor with section/tool hashes.
            tag: Version tag or commit hash. Use "latest" for most recent,
                "prod" for production, or a commit hash for pinned versions.

        Returns:
            PromptOverride if found and hashes match, None otherwise.
        """
        repo_name = self._repo_name(descriptor.ns, descriptor.key)
        prompt_id = f"{repo_name}:{tag}" if tag != "latest" else repo_name

        try:
            prompt = self._client.pull_prompt(prompt_id)
        except Exception:
            return None  # Prompt not found in Hub

        # Extract metadata for hash validation
        metadata = getattr(prompt, "metadata", {}) or {}
        wink_hashes = metadata.get("wink_hashes", {})

        # Build section overrides from prompt template variables
        sections: dict[tuple[str, ...], SectionOverride] = {}
        input_variables = getattr(prompt, "input_variables", []) or []

        for section_desc in descriptor.sections:
            section_key = section_desc.path[-1] if section_desc.path else ""
            var_name = f"{section_key}_section"

            if var_name in input_variables:
                stored_hash = wink_hashes.get(section_key)
                if stored_hash and stored_hash != section_desc.content_hash:
                    # Hash mismatch - section template changed, skip override
                    continue

                # Extract section content from prompt template
                template_content = self._extract_section(prompt, var_name)
                if template_content:
                    sections[section_desc.path] = SectionOverride(
                        expected_hash=section_desc.content_hash,
                        body=template_content,
                    )

        # Build tool overrides from metadata
        tool_overrides: dict[str, ToolOverride] = {}
        tool_metadata = metadata.get("wink_tools", {})

        for tool_desc in descriptor.tools:
            tool_data = tool_metadata.get(tool_desc.name)
            if tool_data:
                stored_hash = tool_data.get("contract_hash")
                if stored_hash and stored_hash != tool_desc.contract_hash:
                    continue  # Contract changed, skip override

                tool_overrides[tool_desc.name] = ToolOverride(
                    name=tool_desc.name,
                    expected_contract_hash=tool_desc.contract_hash,
                    description=tool_data.get("description"),
                    param_descriptions=tool_data.get("param_descriptions", {}),
                )

        if not sections and not tool_overrides:
            return None

        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tool_overrides,
        )

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        """Push prompt override to LangSmith Hub.

        Creates or updates a prompt in the Hub with WINK metadata for
        hash validation on subsequent pulls.
        """
        from langchain_core.prompts import ChatPromptTemplate

        repo_name = self._repo_name(descriptor.ns, descriptor.key)

        # Build template with section variables
        template_parts: list[str] = []
        input_variables: list[str] = []
        wink_hashes: dict[str, str] = {}

        for section_desc in descriptor.sections:
            section_key = section_desc.path[-1] if section_desc.path else ""
            var_name = f"{section_key}_section"
            template_parts.append(f"{{{var_name}}}")
            input_variables.append(var_name)
            wink_hashes[section_key] = section_desc.content_hash

        template = "\n\n".join(template_parts)
        prompt = ChatPromptTemplate.from_template(template)

        # Store WINK metadata
        metadata = {
            "wink_hashes": wink_hashes,
            "wink_ns": descriptor.ns,
            "wink_key": descriptor.key,
            "wink_tools": {
                tool.name: {
                    "contract_hash": tool.contract_hash,
                    "description": override.tool_overrides.get(tool.name, ToolOverride(
                        name=tool.name,
                        expected_contract_hash=tool.contract_hash,
                    )).description,
                    "param_descriptions": override.tool_overrides.get(
                        tool.name, ToolOverride(
                            name=tool.name,
                            expected_contract_hash=tool.contract_hash,
                        )
                    ).param_descriptions,
                }
                for tool in descriptor.tools
            },
        }

        # Push to Hub with tag
        self._client.push_prompt(
            repo_name,
            object=prompt,
            tags=[override.tag] if override.tag != "latest" else None,
        )

        return override

    def delete(self, *, ns: str, prompt_key: str, tag: str) -> None:
        """Delete is not supported by LangSmith Hub.

        LangSmith maintains version history; use tags to deprecate versions.
        """
        raise NotImplementedError(
            "LangSmith Hub does not support deletion. "
            "Remove the tag or archive the prompt in the LangSmith UI."
        )

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Initialize a Hub prompt from an existing WINK prompt."""
        descriptor = descriptor_for_prompt(prompt)
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections={},
            tool_overrides={},
        )
        return self.upsert(descriptor, override)

    @staticmethod
    def _extract_section(prompt: object, var_name: str) -> str | None:
        """Extract section content from a LangSmith prompt template."""
        # Implementation depends on prompt structure
        # May need to parse template or access partial variables
        template = getattr(prompt, "template", None)
        if template and f"{{{var_name}}}" in template:
            # Section is a variable - return None to use default
            return None
        partial_variables = getattr(prompt, "partial_variables", {}) or {}
        return partial_variables.get(var_name)
```

## Version Pinning for Production

For production deployments, pin prompt versions using commit hashes:

```python
# Development: always use latest
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="latest")
)

# Staging: use a named tag
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="staging")
)

# Production: pin to commit hash
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="a1b2c3d4")  # Specific commit
)

# Or use the "prod" tag that you update deliberately
store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(default_tag="prod")
)
```

## Tracing Integration

The adapter automatically integrates with LangSmith tracing when environment
variables are configured. For custom tracing, use the `@traceable` decorator:

```python
from langsmith import traceable

@traceable(run_type="chain", name="WINK Agent Execution")
def execute_with_tracing(
    adapter: LangGraphAdapter,
    prompt: Prompt[OutputT],
    session: SessionProtocol,
) -> PromptResponse[OutputT]:
    """Execute adapter with custom LangSmith trace metadata."""
    return adapter.evaluate(prompt, session=session)
```

### Callback-Based Tracing

For more control, inject LangSmith callbacks directly:

```python
from langsmith.run_helpers import get_current_run_tree

class LangSmithTracingCallback(BaseCallbackHandler):
    """Callback that enriches LangSmith traces with WINK metadata."""

    def __init__(self, prompt_name: str, session_id: str | None) -> None:
        self.prompt_name = prompt_name
        self.session_id = session_id

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra["wink_prompt"] = self.prompt_name
            run_tree.extra["wink_session"] = self.session_id

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra["wink_tool_input"] = input_str
```

## Usage with LangGraph Adapter

```python
from weakincentives.adapters.langgraph import (
    LangGraphAdapter,
    LangGraphClientConfig,
    LangGraphModelConfig,
)
from weakincentives.integrations.langsmith import (
    LangSmithStoreConfig,
    LangSmithPromptOverridesStore,
)

# Configure LangSmith store
langsmith_store = LangSmithPromptOverridesStore(
    LangSmithStoreConfig(
        organization="myorg",
        default_tag="prod",
    )
)

# Create adapter with LangSmith integration
adapter = LangGraphAdapter(
    model="gpt-4o",
    model_config=LangGraphModelConfig(model_provider="openai"),
)

# Evaluate with LangSmith-managed overrides
response = adapter.evaluate(
    prompt,
    session=session,
    overrides_store=langsmith_store,
    overrides_tag="prod",
)
```

## OpenTelemetry Integration

LangSmith supports OpenTelemetry for cross-service tracing:

```python
from opentelemetry import trace
from langsmith.wrappers import wrap_openai

# Wrap LLM client for OTel traces
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("wink_agent_execution"):
    response = adapter.evaluate(prompt, session=session)
```

## Prompt Hub Workflow

**Development Workflow**:

1. Author prompts in WINK codebase (source of truth)
2. Seed to LangSmith Hub: `store.seed(prompt, tag="dev")`
3. Iterate in LangSmith Playground
4. Pull changes back: `store.resolve(descriptor, tag="dev")`

**Production Workflow**:

1. Test with `staging` tag
2. Promote to `prod` tag in LangSmith UI
3. Application pulls `prod` tag for stable execution
4. Rollback by reverting tag to previous commit hash

## Module Structure

```
src/weakincentives/integrations/
└── langsmith/
    ├── __init__.py
    ├── store.py          # LangSmithPromptOverridesStore
    ├── tracing.py        # Tracing callbacks and decorators
    └── config.py         # LangSmithStoreConfig
```

## Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
langsmith = [
    "langsmith>=0.1.0",
]
```

For full LangGraph + LangSmith integration:

```toml
langgraph = [
    "langgraph>=1.0.0,<2.0",
    "langchain>=1.0.0,<2.0",
    "langchain-core>=1.0.0,<2.0",
    "langsmith>=0.1.0",  # Include for full integration
]
```

## Limitations

- **No deletion support**: LangSmith Hub maintains version history; use tags
  to deprecate versions rather than deleting.
- **Hash validation drift**: If WINK prompts change locally, Hub overrides
  may become stale. Re-seed to synchronize.
- **Organization namespacing**: Prompts are scoped to organizations; ensure
  consistent configuration across environments.

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Manage Prompts Programmatically](https://docs.langchain.com/langsmith/manage-prompts)
- [Trace with LangGraph](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_langgraph)
- [LangChain Hub Announcement](https://blog.langchain.com/langchain-prompt-hub/)
- [End-to-End OpenTelemetry Support](https://blog.langchain.com/end-to-end-opentelemetry-langsmith/)
