# Dataclass serialization audit

This package hosts a pytest plugin that validates whether dataclasses wired into
the session snapshot pipeline round-trip through `weakincentives.serde`.  When a
new runtime or tool module registers reducers for additional dataclasses, extend
the discovery table in [`dataclass_serde.py`](./dataclass_serde.py):

1. Add a helper such as `_discover_<feature>()` that imports the module and
   returns factories for each dataclass the session stores.
2. Append the helper to `_DISCOVERY_BUILDERS` so the plugin parameterises the
   shared `test_dataclass_serialization` case.
3. Provide lightweight factory functions so the test can instantiate each
   dataclass without bespoke fixtures. If a dataclass cannot be constructed with
   simple sample data, decorate it with `@pytest.mark.skip_serialization` to opt
   out until a custom fixture is available.

Keeping the discovery map current ensures coverage stays aligned with the
snapshots persisted during agent runs.
