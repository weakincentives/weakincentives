# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt composition system for building structured AI prompts.

This package provides a declarative framework for composing, rendering, and
customizing prompts for large language models. The architecture separates
prompt definition (templates) from prompt execution (bound prompts), enabling
reuse, testing, and runtime customization.

Core Concepts
-------------

**PromptTemplate**
    An immutable specification of prompt structure. Templates declare:

    - A namespace (``ns``) and key for identification
    - Sections that contribute content and tools
    - Policies for tool invocation ordering
    - Optional structured output schema
    - Resource bindings for dependency injection

**Prompt**
    A bound prompt ready for rendering. Created by wrapping a template with
    runtime configuration (overrides store, parameters, resources). The prompt
    is the only way to render a template.

**Section**
    The building block for prompt content. Sections:

    - Render markdown content with optional parameter substitution
    - Declare tools available to the model
    - Support hierarchical nesting via ``children``
    - Control visibility (FULL, SUMMARY, HIDDEN)
    - Accept parameter overrides at runtime

**Tool**
    A callable exposed to the model. Tools have:

    - A typed parameter schema (dataclass)
    - A typed result schema (dataclass or None)
    - A handler function implementing the tool logic
    - Optional examples for few-shot guidance

**ToolResult**
    The structured response from a tool handler. Supports success/error states
    and typed payloads that render to text for the model.

Section Key Naming
------------------

Section keys must match the pattern: ``^[a-z0-9][a-z0-9._-]{0,63}$``

- Start with a lowercase letter or digit
- Contain only lowercase letters, digits, dots, underscores, or hyphens
- Maximum 64 characters

Tool names follow OpenAI function naming constraints: lowercase ASCII letters,
digits, underscores, or hyphens (1-64 characters).

Visibility System
-----------------

Sections support three visibility levels:

- **FULL**: Render complete section content
- **SUMMARY**: Render abbreviated content (requires ``summary`` template)
- **HIDDEN**: Exclude section from rendered output

Visibility can be controlled via:

1. Section constructor defaults
2. Session state overrides (VisibilityOverrides)
3. Callable selectors that inspect parameters or session

Progressive disclosure tools (``open_sections``, ``read_section``) are
automatically injected when sections are rendered with SUMMARY visibility.

Tool Policies
-------------

Policies enforce constraints on tool invocation order:

- **SequentialDependencyPolicy**: Require tools to be called in order
- **ReadBeforeWritePolicy**: Require files to be read before overwriting

Policies track state in the session and block tool calls that violate
constraints, returning an error to the model.

Resource Management
-------------------

Resources are managed via the prompt's resource context::

    prompt = Prompt(template).bind(params, resources={Filesystem: fs})

    with prompt.resources:
        rendered = prompt.render()
        filesystem = prompt.resources.get(Filesystem)
    # Resources cleaned up

Resources can be declared at template, section, or bind-time levels.
Later declarations override earlier ones.

Basic Usage
-----------

Creating a prompt template::

    from dataclasses import dataclass
    from weakincentives.prompt import (
        PromptTemplate,
        Section,
        Tool,
        ToolContext,
        ToolResult,
    )

    @dataclass(slots=True, frozen=True)
    class TaskParams:
        objective: str
        deadline: str

    @dataclass(slots=True, frozen=True)
    class GreetParams:
        name: str

    @dataclass(slots=True, frozen=True)
    class GreetResult:
        message: str

        def render(self) -> str:
            return self.message

    def greet_handler(
        params: GreetParams,
        *,
        context: ToolContext,
    ) -> ToolResult[GreetResult]:
        \"\"\"Greet a user by name.\"\"\"
        return ToolResult.ok(
            GreetResult(message=f"Hello, {params.name}!"),
            message="Greeted successfully",
        )

    class InstructionsSection(Section[TaskParams]):
        def __init__(self) -> None:
            super().__init__(
                title="Instructions",
                key="instructions",
                tools=(
                    Tool[GreetParams, GreetResult](
                        name="greet",
                        description="Greet a user by name",
                        handler=greet_handler,
                    ),
                ),
            )

        def render_body(
            self,
            params: TaskParams | None,
            *,
            visibility=None,
            path=(),
            session=None,
        ) -> str:
            if params is None:
                return "No task specified."
            return f"Complete: {params.objective} by {params.deadline}"

        def clone(self, **kwargs):
            return InstructionsSection()

    template = PromptTemplate(
        ns="example",
        key="task-prompt",
        sections=[InstructionsSection()],
    )

Rendering a prompt::

    from weakincentives.prompt import Prompt

    prompt = Prompt(template)
    prompt.bind(TaskParams(objective="Write tests", deadline="Friday"))

    with prompt.resources:
        rendered = prompt.render()
        print(rendered.text)
        print(f"Tools: {[t.name for t in rendered.tools]}")

Using structured output::

    @dataclass(slots=True, frozen=True)
    class TaskOutput:
        status: str
        summary: str

    template = PromptTemplate[TaskOutput](
        ns="example",
        key="structured-task",
        sections=[...],
    )

Overrides System
----------------

The :mod:`weakincentives.prompt.overrides` subpackage provides runtime
customization of prompts without code changes. Overrides support:

- Section body replacement (with hash-based staleness detection)
- Tool description patching
- Task example modification

See :mod:`weakincentives.prompt.overrides` for details.

Feedback System
---------------

Feedback providers enable runtime observations during prompt execution:

- **DeadlineFeedback**: Inject time remaining information
- Custom providers via ``FeedbackProvider`` protocol

Feedback is collected via ``run_feedback_providers()`` and integrated
into the prompt context.

Module Structure
----------------

- ``prompt``: PromptTemplate and Prompt classes
- ``section``: Section base class and visibility
- ``tool``: Tool, ToolContext, ToolHandler, ToolExample
- ``tool_result``: ToolResult container
- ``rendering``: PromptRenderer and RenderedPrompt
- ``registry``: Section registration and validation
- ``policy``: ToolPolicy protocol and implementations
- ``protocols``: Structural typing interfaces
- ``overrides``: Runtime customization (subpackage)
- ``feedback``: Feedback provider system
- ``markdown``: MarkdownSection for template-based sections
- ``task_examples``: TaskExample and TaskExamplesSection
"""

from __future__ import annotations

from ._prompt_resources import PromptResources
from ._types import ToolRenderableResult
from .errors import (
    PromptError,
    PromptRenderError,
    PromptValidationError,
    SectionPath,
    VisibilityExpansionRequired,
)
from .feedback import (
    Feedback,
    FeedbackContext,
    FeedbackProvider,
    FeedbackProviderConfig,
    FeedbackTrigger,
    Observation,
    collect_feedback,
    run_feedback_providers,
)
from .feedback_providers import DeadlineFeedback
from .markdown import MarkdownSection
from .overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
    hash_json,
    hash_text,
)
from .policy import (
    PolicyDecision,
    PolicyState,
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
    ToolPolicy,
)
from .progressive_disclosure import (
    OpenSectionsParams,
    ReadSectionParams,
    ReadSectionResult,
)
from .prompt import Prompt, PromptTemplate, SectionNode
from .protocols import (
    PromptProtocol,
    PromptTemplateProtocol,
    ProviderAdapterProtocol,
    RenderedPromptProtocol,
    ToolSuiteSection,
    WorkspaceSection,
)
from .rendering import RenderedPrompt
from .section import Section, SectionVisibility
from .structured_output import (
    OutputParseError,
    StructuredOutputConfig,
    parse_structured_output,
)
from .task_examples import TaskExample, TaskExamplesSection, TaskStep
from .tool import Tool, ToolContext, ToolExample, ToolHandler
from .tool_result import ToolResult

__all__ = [
    "DeadlineFeedback",
    "Feedback",
    "FeedbackContext",
    "FeedbackProvider",
    "FeedbackProviderConfig",
    "FeedbackTrigger",
    "LocalPromptOverridesStore",
    "MarkdownSection",
    "Observation",
    "OpenSectionsParams",
    "OutputParseError",
    "PolicyDecision",
    "PolicyState",
    "Prompt",
    "PromptDescriptor",
    "PromptError",
    "PromptLike",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "PromptProtocol",
    "PromptRenderError",
    "PromptResources",
    "PromptTemplate",
    "PromptTemplateProtocol",
    "PromptValidationError",
    "ProviderAdapterProtocol",
    "ReadBeforeWritePolicy",
    "ReadSectionParams",
    "ReadSectionResult",
    "RenderedPrompt",
    "RenderedPromptProtocol",
    "Section",
    "SectionDescriptor",
    "SectionNode",
    "SectionOverride",
    "SectionPath",
    "SectionVisibility",
    "SequentialDependencyPolicy",
    "StructuredOutputConfig",
    "TaskExample",
    "TaskExamplesSection",
    "TaskStep",
    "Tool",
    "ToolContext",
    "ToolDescriptor",
    "ToolExample",
    "ToolHandler",
    "ToolOverride",
    "ToolPolicy",
    "ToolRenderableResult",
    "ToolResult",
    "ToolSuiteSection",
    "VisibilityExpansionRequired",
    "WorkspaceSection",
    "collect_feedback",
    "hash_json",
    "hash_text",
    "parse_structured_output",
    "run_feedback_providers",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
