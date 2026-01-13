# Security Assessment Report

**Date:** 2026-01-13
**Scope:** weakincentives codebase (comprehensive review)
**Classification:** Security Assessment

## Executive Summary

The weakincentives codebase demonstrates **strong security discipline** with
defense-in-depth patterns throughout. No critical vulnerabilities were
identified. The architecture follows secure-by-default principles with
sandboxed execution environments, strict input validation, and proper
credential handling.

**Overall Security Posture: GOOD**

---

## 1. Automated Security Scanning Results

### 1.1 Bandit (Static Analysis)
- **Status:** PASSED
- **Findings:** No security issues detected
- **Notes:** Several intentional subprocess usages are marked with `# nosec`
  comments with proper justification (sandboxed Podman execution, git operations)

### 1.2 pip-audit (Dependency Vulnerabilities)
- **Status:** PASSED
- **Findings:** No known vulnerabilities in dependencies
- **Dependencies reviewed:** pyyaml, asteval, openai, litellm, podman, redis,
  claude-agent-sdk, fastapi, uvicorn

---

## 2. Sandbox Implementations

### 2.1 Podman Container Sandbox (`contrib/tools/podman.py`)

**Security Rating: STRONG**

| Control | Implementation | Assessment |
|---------|---------------|------------|
| Network isolation | `network_mode="none"` | Prevents all network access |
| Memory limits | `mem_limit="1g"`, `memswap_limit="1g"` | Prevents resource exhaustion |
| CPU limits | `cpu_period=100_000`, `cpu_quota=100_000` | Prevents CPU monopolization |
| User isolation | `user="65534:65534"` (nobody) | Non-root execution |
| Filesystem | Read-only overlay, tmpfs for `/tmp` | Prevents persistent changes |

**Input Validation:**
- Command length: Max 4,096 characters
- ASCII-only enforcement for all inputs
- Environment variables: Max 64 vars, 80-char keys, 512-char values
- Path traversal blocked: Rejects `.` and `..` segments
- Timeout clamping: 1-120 seconds
- NaN validation for numeric inputs

**Path Security (`_assert_within_overlay`):**
```python
def _assert_within_overlay(self, path: Path) -> None:
    candidate = path.resolve()  # Symlink-safe resolution
    try:
        _ = candidate.relative_to(self._overlay_path)
    except ValueError:
        raise ToolValidationError("Path escapes overlay")
```

### 2.2 Asteval Python Sandbox (`contrib/tools/asteval.py`)

**Security Rating: STRONG**

**Safe Globals Whitelist:**
```python
_SAFE_GLOBALS = {
    "abs", "len", "min", "max", "print", "range", "round", "sum", "str",
    "math", "statistics"  # Read-only MappingProxyType
}
```

**Blocked Operations:**
- No `import`, `__import__`, `exec`, `eval`
- No file I/O, network access, or system calls
- Node handlers for dangerous operations are removed at runtime

**Constraints:**
- Code size: Max 2,000 characters
- Timeout: 5 seconds
- Output truncation: 4,096 characters
- Control character filtering (only `\n`, `\t` allowed)

### 2.3 Virtual Filesystem (`contrib/tools/vfs.py`, `filesystem/_host.py`)

**Security Rating: STRONG**

**Path Resolution & Escape Prevention:**
```python
def _resolve_path(self, path: str) -> Path:
    root_path = Path(self._root).resolve()
    candidate = (root_path / path).resolve()  # Handles .. segments
    _ = candidate.relative_to(root_path)  # Escape detection
    return candidate
```

**Security Features:**
- Symlink-safe: `.resolve()` follows symlinks before boundary check
- TOCTOU-resistant: Atomic boundary check after resolution
- Git operations isolated: External `--git-dir` prevents repo tampering
- GIT_* environment variables filtered to prevent hook injection

---

## 3. Serialization/Deserialization

**Security Rating: STRONG**

### 3.1 Safe Patterns Used
- **No pickle, marshal, or shelve:** Only JSON-based serialization
- **No eval/exec:** Type coercion uses safe constructors
- **YAML:** Only `yaml.safe_load()` used (in skills validation)
- **Type validation:** Strict dataclass parsing with type hints

### 3.2 Serde Implementation (`serde/parse.py`)
- Maps JSON input to dataclass instances
- Field type validation via `get_type_hints()`
- Extra fields handling: configurable (`ignore`, `forbid`, `allow`)
- Validation hooks: `__validate__()`, `__post_validate__()`

---

## 4. Authentication & Secret Handling

**Security Rating: GOOD**

### 4.1 API Key Management
- API keys read from environment variables (not hardcoded)
- Delegated to provider SDKs (OpenAI, LiteLLM, Claude)
- No credential logging observed

### 4.2 Environment Variable Filtering (`adapters/claude_agent_sdk/isolation.py`)
```python
_SENSITIVE_ENV_PREFIXES = (
    "HOME", "CLAUDE_", "ANTHROPIC_", "AWS_",
    "GOOGLE_", "AZURE_", "OPENAI_"
)
```

Sensitive variables are explicitly filtered when `include_host_env=True` to
prevent credential leakage to isolated environments.

---

## 5. Injection Vulnerability Analysis

### 5.1 Command Injection
**Risk: LOW**

- Subprocess calls use list-based arguments (not shell strings)
- All subprocess usages marked with `# nosec` and proper justification
- Commands validated: length, ASCII, no shell metacharacters

### 5.2 SQL Injection
**Risk: N/A**

- No SQL database usage in the codebase
- Redis operations use parameterized Lua scripts

### 5.3 Template Injection
**Risk: LOW**

- Uses f-strings and `.format()` for string interpolation
- No user-controlled template engines
- Prompt templates use safe `$param` substitution

### 5.4 Redis Injection
**Risk: LOW**

- Lua scripts use parameterized `KEYS[]` and `ARGV[]`
- No string concatenation in Redis commands

---

## 6. Cryptographic Usage

**Security Rating: ACCEPTABLE**

| Usage | Implementation | Assessment |
|-------|---------------|------------|
| UUIDs | `uuid4()` | Cryptographically secure |
| Hashing | `hashlib.sha256` | Appropriate for checksums |
| Random | `random.uniform` for jitter | Non-security use (marked `# nosec B311`) |

**Note:** No encryption or authentication tokens implemented directly;
delegated to provider SDKs.

---

## 7. Thread Safety

**Security Rating: STRONG**

The codebase demonstrates proper concurrent programming patterns:

- `threading.Lock` for mutual exclusion
- `threading.RLock` for reentrant locking (session, transactions)
- `threading.Condition` for producer/consumer patterns (mailbox)
- `threading.Event` for shutdown coordination
- Callbacks invoked outside locks to prevent deadlock

Example from `Heartbeat`:
```python
def beat(self) -> None:
    with self._lock:
        self._last_beat = time.monotonic()
        callbacks = list(self._callbacks)  # Snapshot under lock
    # Invoke outside lock to avoid deadlock
    for callback in callbacks:
        callback()
```

---

## 8. Network & HTTP Security

**Security Rating: GOOD**

- No direct HTTP client usage; delegated to provider SDKs
- Network policy support for Claude Agent SDK sandbox
- Default `network_mode="none"` in Podman containers
- Health endpoints on configurable port (Kubernetes probes)

---

## 9. OWASP Top 10 Analysis

| Category | Status | Notes |
|----------|--------|-------|
| A01: Broken Access Control | N/A | Library, not web app |
| A02: Cryptographic Failures | PASS | Proper use of hashlib, uuid4 |
| A03: Injection | PASS | Parameterized commands, no SQL |
| A04: Insecure Design | PASS | Defense-in-depth architecture |
| A05: Security Misconfiguration | PASS | Secure defaults, explicit nosec |
| A06: Vulnerable Components | PASS | pip-audit clean |
| A07: Auth Failures | N/A | Delegated to provider SDKs |
| A08: Software Integrity | PASS | No deserialization of untrusted code |
| A09: Logging Failures | PASS | Structured logging, no secrets |
| A10: SSRF | N/A | No URL fetching |

---

## 10. Security Recommendations

### 10.1 Low Priority (Enhancements)

1. **Document nosec justifications:** Some `# nosec` comments could benefit
   from more detailed justification (e.g., `B603` in `formal/testing.py`)

2. **Rate limiting documentation:** Document recommended rate limits for
   health endpoints when exposed publicly

3. **Dependency pinning:** Consider pinning exact versions in production
   deployments to prevent supply chain attacks

### 10.2 Best Practices Already Implemented

- Design-by-contract (`@require`, `@ensure`, `@invariant`)
- 100% test coverage requirement
- Pre-commit hooks for security scanning
- Explicit sensitive environment variable filtering
- Immutable dataclasses (`frozen=True`, `slots=True`)

---

## 11. Files Reviewed

| File | Focus Area |
|------|-----------|
| `contrib/tools/podman.py` | Container sandbox, input validation |
| `contrib/tools/asteval.py` | Python sandbox, safe globals |
| `contrib/tools/vfs.py` | Virtual filesystem, path handling |
| `filesystem/_host.py` | Host filesystem, path escape prevention |
| `serde/parse.py` | Deserialization, type coercion |
| `serde/dump.py` | Serialization |
| `adapters/claude_agent_sdk/isolation.py` | Hermetic isolation, env filtering |
| `adapters/openai.py` | API client, credential handling |
| `contrib/mailbox/_redis.py` | Redis operations, Lua scripts |
| `runtime/watchdog.py` | Thread safety, health monitoring |
| `skills/_validation.py` | YAML parsing, skill validation |

---

## 12. Conclusion

The weakincentives codebase exhibits mature security practices:

- **Sandboxing:** Multiple isolation layers (Podman, asteval, VFS)
- **Input validation:** Comprehensive validation at all boundaries
- **Credential management:** Proper delegation and filtering
- **Thread safety:** Correct synchronization primitives
- **Dependency management:** Clean vulnerability scans

No critical or high-severity vulnerabilities were identified. The codebase
is suitable for use in production environments with appropriate operational
security controls.

---

*This assessment was performed using automated tools (bandit, pip-audit)
and manual code review.*
