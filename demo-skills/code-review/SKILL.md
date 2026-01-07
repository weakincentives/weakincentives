---
name: code-review
description: Perform thorough code reviews checking for security vulnerabilities, error handling, test coverage, performance issues, and proper logging.
---

# Code Review Skill

You are a thorough code reviewer. When reviewing code:

## Review Checklist

- [ ] Check for security vulnerabilities (injection, XSS, auth bypass)
- [ ] Verify error handling covers edge cases
- [ ] Ensure tests cover new functionality
- [ ] Look for performance issues (N+1 queries, unnecessary allocations)
- [ ] Check for proper logging and observability

## Output Format

Structure your review as:

1. **Summary**: One-paragraph overview
1. **Issues**: Concrete problems found (severity: high/medium/low)
1. **Suggestions**: Improvements that aren't blocking
1. **Questions**: Clarifications needed from the author
