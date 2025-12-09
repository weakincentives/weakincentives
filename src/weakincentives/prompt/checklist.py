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

"""Checklist section for domain-specific review checklists with progressive disclosure."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, cast, override

from ._types import SupportsDataclass
from ._visibility import SectionVisibility, VisibilitySelector
from .section import Section


@dataclass(slots=True, frozen=True)
class ChecklistItem:
    """A single item in a checklist.

    Attributes:
        text: The checklist item text to display.
        category: Optional category grouping for the item.
        severity: Importance level (critical, high, medium, low).
    """

    text: str
    category: str | None = None
    severity: str = "medium"


@dataclass(slots=True, frozen=True)
class ChecklistParams:
    """Parameters for checklist section rendering.

    Attributes:
        domain: The domain name for the checklist (e.g., "security", "performance").
        item_count: Number of items in the checklist, auto-populated during rendering.
    """

    domain: str = field(
        default="",
        metadata={"description": "Domain name for the checklist."},
    )
    item_count: int = field(
        default=0,
        metadata={"description": "Number of items in the checklist."},
    )


class ChecklistSection(Section[ChecklistParams]):
    """Section that renders domain-specific review checklists.

    This section extends the base Section to provide specialized rendering
    for checklist-style content with progressive disclosure support. In
    SUMMARY mode, it shows a brief overview with item count. In FULL mode,
    it renders all checklist items organized by category.

    Example:
        >>> items = [
        ...     ChecklistItem("Check for SQL injection", category="Input Validation"),
        ...     ChecklistItem("Validate user permissions", category="Authorization"),
        ... ]
        >>> section = ChecklistSection(
        ...     title="Security Checklist",
        ...     key="security-checklist",
        ...     domain="security",
        ...     items=items,
        ...     preamble="Review these security considerations:",
        ...     visibility=SectionVisibility.SUMMARY,
        ... )
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        title: str,
        key: str,
        domain: str,
        items: Sequence[ChecklistItem],
        preamble: str = "",
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a checklist section.

        Args:
            title: Display title for the section.
            key: Unique identifier for the section.
            domain: Domain name (e.g., "security", "performance").
            items: Sequence of checklist items to render.
            preamble: Optional introductory text before the checklist.
            children: Optional nested sections.
            enabled: Optional predicate to conditionally enable the section.
            tools: Optional tools exposed by this section.
            accepts_overrides: Whether the section accepts prompt overrides.
            visibility: Default visibility (FULL or SUMMARY).
        """
        self._domain = domain
        self._items = tuple(items)
        self._preamble = preamble.strip()

        # Build the summary text
        item_count = len(self._items)
        categories = {item.category for item in self._items if item.category}
        category_hint = (
            f" ({len(categories)} categories)" if len(categories) > 1 else ""
        )
        summary = (
            f"{item_count} {domain} review items available{category_hint}. "
            "Request expansion for detailed checklist."
        )

        default_params = ChecklistParams(domain=domain, item_count=item_count)

        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @property
    def domain(self) -> str:
        """Return the checklist domain."""
        return self._domain

    @property
    def items(self) -> tuple[ChecklistItem, ...]:
        """Return the checklist items."""
        return self._items

    @property
    def preamble(self) -> str:
        """Return the preamble text."""
        return self._preamble

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        path: tuple[str, ...] = (),
        visibility: SectionVisibility | None = None,
    ) -> str:
        """Render the checklist section.

        In SUMMARY mode, renders a brief overview with item count.
        In FULL mode, renders all items organized by category.
        """
        effective = self.effective_visibility(override=visibility, params=params)
        heading = self._render_heading(depth, number, path)

        if effective == SectionVisibility.SUMMARY and self.summary is not None:
            return f"{heading}\n\n{self.summary}"

        return f"{heading}\n\n{self._render_full_body()}"

    def _render_heading(self, depth: int, number: str, path: tuple[str, ...]) -> str:
        """Render the section heading."""
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        path_str = ".".join(path) if path else ""
        title_with_path = (
            f"{self.title.strip()} ({path_str})" if path_str else self.title.strip()
        )
        return f"{heading_level} {normalized_number}. {title_with_path}"

    def _render_full_body(self) -> str:
        """Render the full checklist body with items grouped by category."""
        lines: list[str] = []

        if self._preamble:
            lines.append(self._preamble)
            lines.append("")

        # Group items by category
        categorized: dict[str | None, list[ChecklistItem]] = {}
        for item in self._items:
            category = item.category
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)

        # Render items by category
        for category, category_items in categorized.items():
            if category is not None:
                lines.append(f"**{category}**")
                lines.append("")

            for item in category_items:
                severity_marker = self._severity_marker(item.severity)
                lines.append(f"- [ ] {severity_marker}{item.text}")

            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _severity_marker(severity: str) -> str:
        """Return a marker prefix based on severity level."""
        markers = {
            "critical": "[CRITICAL] ",
            "high": "[HIGH] ",
            "medium": "",
            "low": "[LOW] ",
        }
        return markers.get(severity.lower(), "")

    @override
    def original_body_template(self) -> str | None:
        """Return a template representation for hashing."""
        # Build a stable template from items
        item_texts = [f"- {item.text}" for item in self._items]
        return "\n".join([self._preamble, *item_texts])

    @override
    def clone(self, **kwargs: object) -> Self:
        """Return a deep copy of the section."""
        cloned_children: list[Section[SupportsDataclass]] = []
        for child in self.children:
            if not hasattr(child, "clone"):
                raise TypeError(
                    "Section children must implement clone()."
                )  # pragma: no cover
            cloned_children.append(child.clone(**kwargs))

        cls: type[Any] = type(self)
        clone = cls(
            title=self.title,
            key=self.key,
            domain=self._domain,
            items=self._items,
            preamble=self._preamble,
            children=cloned_children,
            enabled=self._enabled,
            tools=self.tools(),
            accepts_overrides=self.accepts_overrides,
            visibility=self.visibility,
        )
        return cast(Self, clone)


# =============================================================================
# Domain-Specific Checklist Builders
# =============================================================================


def build_security_checklist() -> ChecklistSection:
    """Build a security review checklist based on OWASP Top 10.

    Returns:
        A ChecklistSection configured for security reviews with items
        covering injection, authentication, data exposure, and more.
    """
    items = [
        # Injection
        ChecklistItem(
            "Verify parameterized queries for all database operations",
            category="Injection Prevention",
            severity="critical",
        ),
        ChecklistItem(
            "Check for command injection in shell/system calls",
            category="Injection Prevention",
            severity="critical",
        ),
        ChecklistItem(
            "Validate LDAP/XPath/NoSQL query construction",
            category="Injection Prevention",
            severity="high",
        ),
        # Authentication
        ChecklistItem(
            "Ensure password hashing uses bcrypt/argon2 with proper cost",
            category="Authentication",
            severity="critical",
        ),
        ChecklistItem(
            "Verify multi-factor authentication for sensitive operations",
            category="Authentication",
            severity="high",
        ),
        ChecklistItem(
            "Check session token generation uses cryptographically secure randomness",
            category="Authentication",
            severity="high",
        ),
        ChecklistItem(
            "Validate session timeout and invalidation on logout",
            category="Authentication",
            severity="medium",
        ),
        # Data Exposure
        ChecklistItem(
            "Confirm sensitive data encrypted at rest (AES-256 or equivalent)",
            category="Data Protection",
            severity="critical",
        ),
        ChecklistItem(
            "Verify TLS 1.2+ for all data in transit",
            category="Data Protection",
            severity="critical",
        ),
        ChecklistItem(
            "Check for accidental logging of sensitive data (PII, credentials)",
            category="Data Protection",
            severity="high",
        ),
        ChecklistItem(
            "Validate secrets management (no hardcoded credentials)",
            category="Data Protection",
            severity="critical",
        ),
        # Access Control
        ChecklistItem(
            "Verify authorization checks on all protected endpoints",
            category="Access Control",
            severity="critical",
        ),
        ChecklistItem(
            "Check for IDOR vulnerabilities in resource access",
            category="Access Control",
            severity="high",
        ),
        ChecklistItem(
            "Validate principle of least privilege in role assignments",
            category="Access Control",
            severity="medium",
        ),
        # Security Misconfiguration
        ChecklistItem(
            "Ensure security headers present (CSP, X-Frame-Options, etc.)",
            category="Security Configuration",
            severity="high",
        ),
        ChecklistItem(
            "Check for exposed debug endpoints or verbose error messages",
            category="Security Configuration",
            severity="high",
        ),
        ChecklistItem(
            "Verify default credentials changed and unnecessary features disabled",
            category="Security Configuration",
            severity="medium",
        ),
        # XSS
        ChecklistItem(
            "Validate output encoding for all user-controlled content",
            category="XSS Prevention",
            severity="critical",
        ),
        ChecklistItem(
            "Check for DOM-based XSS in client-side JavaScript",
            category="XSS Prevention",
            severity="high",
        ),
        # Deserialization
        ChecklistItem(
            "Avoid deserializing untrusted data or use safe alternatives",
            category="Deserialization",
            severity="critical",
        ),
        # Logging & Monitoring
        ChecklistItem(
            "Ensure security events are logged (auth failures, access denials)",
            category="Logging & Monitoring",
            severity="medium",
        ),
    ]

    return ChecklistSection(
        title="Security Review Checklist",
        key="checklist.security",
        domain="security",
        items=items,
        preamble=textwrap.dedent(
            """
            Review the code against these security criteria based on OWASP Top 10.
            Mark items as verified or flag concerns for follow-up.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


def build_performance_checklist() -> ChecklistSection:
    """Build a performance review checklist.

    Returns:
        A ChecklistSection configured for performance reviews covering
        database queries, memory management, caching, and concurrency.
    """
    items = [
        # Database Performance
        ChecklistItem(
            "Check for N+1 query patterns in ORM usage",
            category="Database Queries",
            severity="critical",
        ),
        ChecklistItem(
            "Verify indexes exist for frequent query predicates",
            category="Database Queries",
            severity="high",
        ),
        ChecklistItem(
            "Review query plans for expensive operations (full scans, sorts)",
            category="Database Queries",
            severity="high",
        ),
        ChecklistItem(
            "Check for unbounded queries (missing LIMIT clauses)",
            category="Database Queries",
            severity="high",
        ),
        ChecklistItem(
            "Validate connection pooling configuration",
            category="Database Queries",
            severity="medium",
        ),
        # Memory Management
        ChecklistItem(
            "Check for memory leaks in long-running processes",
            category="Memory Management",
            severity="critical",
        ),
        ChecklistItem(
            "Verify large collections are processed in batches/streams",
            category="Memory Management",
            severity="high",
        ),
        ChecklistItem(
            "Check for circular references preventing garbage collection",
            category="Memory Management",
            severity="medium",
        ),
        ChecklistItem(
            "Review buffer sizes for I/O operations",
            category="Memory Management",
            severity="medium",
        ),
        # Caching
        ChecklistItem(
            "Identify cacheable operations (expensive computations, remote calls)",
            category="Caching",
            severity="medium",
        ),
        ChecklistItem(
            "Verify cache invalidation strategy is correct",
            category="Caching",
            severity="high",
        ),
        ChecklistItem(
            "Check cache TTLs align with data freshness requirements",
            category="Caching",
            severity="medium",
        ),
        # Concurrency
        ChecklistItem(
            "Verify thread-safe access to shared mutable state",
            category="Concurrency",
            severity="critical",
        ),
        ChecklistItem(
            "Check for deadlock potential in lock ordering",
            category="Concurrency",
            severity="high",
        ),
        ChecklistItem(
            "Review async/await patterns for blocking operations",
            category="Concurrency",
            severity="high",
        ),
        ChecklistItem(
            "Validate thread pool sizing for workload characteristics",
            category="Concurrency",
            severity="medium",
        ),
        # Network & I/O
        ChecklistItem(
            "Check for appropriate timeouts on external calls",
            category="Network & I/O",
            severity="high",
        ),
        ChecklistItem(
            "Verify retry logic with exponential backoff",
            category="Network & I/O",
            severity="medium",
        ),
        ChecklistItem(
            "Review payload sizes for API calls (compression, pagination)",
            category="Network & I/O",
            severity="medium",
        ),
        # Algorithms
        ChecklistItem(
            "Verify algorithm complexity is appropriate for data scale",
            category="Algorithms",
            severity="high",
        ),
        ChecklistItem(
            "Check for unnecessary object allocations in hot paths",
            category="Algorithms",
            severity="medium",
        ),
    ]

    return ChecklistSection(
        title="Performance Review Checklist",
        key="checklist.performance",
        domain="performance",
        items=items,
        preamble=textwrap.dedent(
            """
            Review the code for performance concerns. Focus on database access patterns,
            memory usage, caching opportunities, and concurrency correctness.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


def build_api_checklist() -> ChecklistSection:
    """Build an API review checklist.

    Returns:
        A ChecklistSection configured for API reviews covering breaking changes,
        versioning, documentation, and error handling.
    """
    items = [
        # Breaking Changes
        ChecklistItem(
            "Check for removed or renamed endpoints",
            category="Breaking Changes",
            severity="critical",
        ),
        ChecklistItem(
            "Verify no required fields added to request schemas",
            category="Breaking Changes",
            severity="critical",
        ),
        ChecklistItem(
            "Check for changed response field types or removal",
            category="Breaking Changes",
            severity="critical",
        ),
        ChecklistItem(
            "Validate HTTP method changes maintain semantics",
            category="Breaking Changes",
            severity="high",
        ),
        ChecklistItem(
            "Review changes to authentication/authorization requirements",
            category="Breaking Changes",
            severity="critical",
        ),
        # Versioning
        ChecklistItem(
            "Verify API version is incremented for breaking changes",
            category="Versioning",
            severity="high",
        ),
        ChecklistItem(
            "Check deprecated endpoints have sunset timeline",
            category="Versioning",
            severity="medium",
        ),
        ChecklistItem(
            "Validate version negotiation works correctly",
            category="Versioning",
            severity="medium",
        ),
        # Request/Response Design
        ChecklistItem(
            "Verify RESTful conventions (resource naming, HTTP verbs)",
            category="API Design",
            severity="medium",
        ),
        ChecklistItem(
            "Check for consistent naming conventions (camelCase/snake_case)",
            category="API Design",
            severity="medium",
        ),
        ChecklistItem(
            "Validate pagination for list endpoints",
            category="API Design",
            severity="high",
        ),
        ChecklistItem(
            "Review response envelope structure consistency",
            category="API Design",
            severity="medium",
        ),
        # Error Handling
        ChecklistItem(
            "Verify appropriate HTTP status codes for error conditions",
            category="Error Handling",
            severity="high",
        ),
        ChecklistItem(
            "Check error response includes actionable details",
            category="Error Handling",
            severity="medium",
        ),
        ChecklistItem(
            "Validate rate limit responses include retry-after",
            category="Error Handling",
            severity="medium",
        ),
        # Documentation
        ChecklistItem(
            "Verify OpenAPI/Swagger spec updated for changes",
            category="Documentation",
            severity="high",
        ),
        ChecklistItem(
            "Check request/response examples are accurate",
            category="Documentation",
            severity="medium",
        ),
        ChecklistItem(
            "Validate changelog entry for API changes",
            category="Documentation",
            severity="medium",
        ),
        # Security (API-specific)
        ChecklistItem(
            "Verify input validation on all request parameters",
            category="API Security",
            severity="high",
        ),
        ChecklistItem(
            "Check for rate limiting on public endpoints",
            category="API Security",
            severity="high",
        ),
        ChecklistItem(
            "Validate CORS configuration is appropriate",
            category="API Security",
            severity="medium",
        ),
    ]

    return ChecklistSection(
        title="API Review Checklist",
        key="checklist.api",
        domain="API",
        items=items,
        preamble=textwrap.dedent(
            """
            Review API changes for backward compatibility, versioning correctness,
            and documentation completeness. Pay special attention to breaking changes.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


def build_test_checklist() -> ChecklistSection:
    """Build a test review checklist.

    Returns:
        A ChecklistSection configured for test reviews covering edge cases,
        mocking patterns, coverage, and test quality.
    """
    items = [
        # Edge Cases
        ChecklistItem(
            "Verify null/None input handling is tested",
            category="Edge Cases",
            severity="high",
        ),
        ChecklistItem(
            "Check empty collection edge cases covered",
            category="Edge Cases",
            severity="high",
        ),
        ChecklistItem(
            "Validate boundary values tested (min, max, overflow)",
            category="Edge Cases",
            severity="high",
        ),
        ChecklistItem(
            "Test concurrent access scenarios if applicable",
            category="Edge Cases",
            severity="medium",
        ),
        ChecklistItem(
            "Verify error/exception paths are tested",
            category="Edge Cases",
            severity="high",
        ),
        # Mocking Patterns
        ChecklistItem(
            "Check mocks verify interaction contracts",
            category="Mocking",
            severity="high",
        ),
        ChecklistItem(
            "Verify external dependencies are isolated",
            category="Mocking",
            severity="high",
        ),
        ChecklistItem(
            "Avoid over-mocking (testing implementation vs behavior)",
            category="Mocking",
            severity="medium",
        ),
        ChecklistItem(
            "Check mock return values match real implementation contracts",
            category="Mocking",
            severity="high",
        ),
        ChecklistItem(
            "Verify time-dependent tests use controlled clocks",
            category="Mocking",
            severity="medium",
        ),
        # Test Structure
        ChecklistItem(
            "Verify test names describe behavior being tested",
            category="Test Quality",
            severity="medium",
        ),
        ChecklistItem(
            "Check tests follow Arrange-Act-Assert pattern",
            category="Test Quality",
            severity="medium",
        ),
        ChecklistItem(
            "Validate tests are independent and repeatable",
            category="Test Quality",
            severity="high",
        ),
        ChecklistItem(
            "Review test data setup for clarity and maintainability",
            category="Test Quality",
            severity="medium",
        ),
        # Coverage
        ChecklistItem(
            "Verify new code has corresponding tests",
            category="Coverage",
            severity="high",
        ),
        ChecklistItem(
            "Check branch coverage for conditional logic",
            category="Coverage",
            severity="high",
        ),
        ChecklistItem(
            "Validate integration tests for cross-component flows",
            category="Coverage",
            severity="medium",
        ),
        # Assertions
        ChecklistItem(
            "Verify assertions are specific (not just 'not null')",
            category="Assertions",
            severity="medium",
        ),
        ChecklistItem(
            "Check for appropriate use of assertion messages",
            category="Assertions",
            severity="low",
        ),
        ChecklistItem(
            "Validate exception assertions check type and message",
            category="Assertions",
            severity="medium",
        ),
        # Performance Tests
        ChecklistItem(
            "Check performance-critical paths have benchmarks",
            category="Performance Testing",
            severity="low",
        ),
    ]

    return ChecklistSection(
        title="Test Review Checklist",
        key="checklist.test",
        domain="testing",
        items=items,
        preamble=textwrap.dedent(
            """
            Review tests for completeness, quality, and correctness. Ensure edge cases
            are covered and mocking patterns follow best practices.
            """
        ).strip(),
        visibility=SectionVisibility.SUMMARY,
    )


__all__ = [
    "ChecklistItem",
    "ChecklistParams",
    "ChecklistSection",
    "build_api_checklist",
    "build_performance_checklist",
    "build_security_checklist",
    "build_test_checklist",
]
