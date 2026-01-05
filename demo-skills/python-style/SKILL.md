# Python Style Skill

Apply Python best practices when reviewing or writing code.

## Style Guidelines

- Follow PEP 8 for formatting
- Use type annotations for all public functions (PEP 484)
- Write docstrings for public APIs (PEP 257)
- Prefer f-strings over .format() or % formatting

## Common Issues to Flag

- Missing type annotations on public functions
- Mutable default arguments (def foo(items=[]))
- Bare except clauses (except: instead of except Exception:)
- Using assert for validation (stripped in optimized mode)

## References

- PEP 8: https://peps.python.org/pep-0008/
- PEP 484: https://peps.python.org/pep-0484/
- PEP 257: https://peps.python.org/pep-0257/
