# Recipes

These are intentionally opinionated patterns that reflect the "weak incentives"
style: reduce surprise, keep state explicit, and make side effects auditable.

## A Code-Review Agent

See `code_reviewer_example.py` in the repo and the
[Code Review Agent](code-review-agent.md) guide for a full, runnable example
that demonstrates:

- Workspace digest + progressive disclosure
- Planning tools
- VFS or Podman sandbox
- Prompt overrides and optimizer hooks
- Structured output review responses

This is the canonical "put it all together" example. Read it after you
understand the individual pieces.

## A Repo Q&A Agent

**Goal**: Answer questions about a codebase quickly.

**Pattern**:

```python nocheck
from weakincentives.contrib.tools import (
    VfsToolsSection,
    VfsConfig,
    WorkspaceDigestSection,
)
from weakincentives.prompt import PromptTemplate, MarkdownSection


def build_qa_template(*, session):
    return PromptTemplate(
        ns="qa",
        key="repo-qa",
        sections=(
            MarkdownSection(
                title="Instructions",
                key="instructions",
                template=(
                    "Answer questions about this repository.\n\n"
                    "Start by reviewing the workspace digest. "
                    "Use grep and read_file to find specific details."
                ),
            ),
            WorkspaceDigestSection(session=session),
            VfsToolsSection(
                session=session,
                config=VfsConfig(
                    mounts=(HostMount(host_path="."),),
                    allowed_host_roots=(".",),
                ),
            ),
            MarkdownSection(
                title="Question",
                key="question",
                template="${question}",
            ),
        ),
    )
```

**Key ideas:**

- Show workspace digest summary by default
- Allow the model to expand it via `read_section`
- Allow VFS `grep`/`glob`/`read_file` for verification
- Token usage stays low for simple questions

## A "Safe Patch" Agent

**Goal**: Generate a patch but avoid uncontrolled writes.

**Pattern**:

```python nocheck
@dataclass(slots=True, frozen=True)
class PatchOutput:
    summary: str
    diff: str  # The patch as unified diff format
    files_changed: tuple[str, ...]


def build_patch_template(*, session):
    return PromptTemplate[PatchOutput](
        ns="patch",
        key="safe-patch",
        sections=(
            MarkdownSection(
                title="Instructions",
                key="instructions",
                template=(
                    "Make the requested change to the codebase.\n\n"
                    "Work in the virtual filesystem. Your changes won't affect "
                    "the real files. When done, output a unified diff that can "
                    "be applied with `patch -p1`."
                ),
            ),
            VfsToolsSection(session=session, config=...),
            MarkdownSection(
                title="Change Request",
                key="request",
                template="${request}",
            ),
        ),
    )
```

**Key ideas:**

- Use VFS tools for edits (writes go to the virtual copy, not the host)
- Require the model to output a diff as structured output
- Optionally run tests in Podman before proposing the patch
- Humans review the diff before applying it to the real repo

## A Research Agent with Progressive Disclosure

**Goal**: Answer deep questions without stuffing a giant blob into the prompt.

**Pattern**:

```python nocheck
def build_research_template(*, session, sources: list[tuple[str, str, str]]):
    # sources is list of (key, title, content)
    source_sections = tuple(
        MarkdownSection(
            title=f"Source: {title}",
            key=key,
            template=content,
            summary=f"{title} is available for reference.",
            visibility=SectionVisibility.SUMMARY,
        )
        for key, title, content in sources
    )

    return PromptTemplate(
        ns="research",
        key="deep-research",
        sections=(
            MarkdownSection(
                title="Instructions",
                key="instructions",
                template=(
                    "Answer the question using the available sources.\n\n"
                    "Sources are summarized. Use open_sections to expand "
                    "relevant sources, or read_section to peek without "
                    "permanently expanding."
                ),
            ),
            *source_sections,
            MarkdownSection(
                title="Question",
                key="question",
                template="${question}",
            ),
        ),
    )
```

**Key ideas:**

- Store sources as summarized sections
- Let the model open only what it needs
- Keep an audit trail via session snapshots
- The session log shows exactly which sources it used

## Common Anti-Patterns

**Stuffing everything into the prompt upfront:**

Instead of including all possible information, use progressive disclosure. The
model can request what it needs.

**Putting business logic in prompt templates:**

Keep complex formatting and logic in Python, where it can be tested. Templates
should be simple substitution.

**Sharing session-bound sections across sessions:**

Each session needs its own tool sections. Use `clone()` or build fresh sections
per session.

**Ignoring tool failures:**

Transactional execution handles rollback, but you should still design tools to
fail gracefully with helpful error messages.

## Next Steps

- [Code Review Agent](code-review-agent.md): The complete worked example
- [Workspace Tools](workspace-tools.md): Understand VFS, Podman, and planning
- [Progressive Disclosure](progressive-disclosure.md): Control context size
