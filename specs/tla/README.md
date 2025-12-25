# Redis Mailbox TLA+ Specification

This directory contains the TLA+ formal specification for the Redis mailbox
implementation. The specification models the mailbox state machine and can be
exhaustively checked by the TLC model checker for safety and liveness properties.

## Files

- `RedisMailbox.tla` - Main state machine specification
- `RedisMailboxMC.tla` - Model checking configuration module
- `RedisMailboxMC.cfg` - TLC configuration file

## Key Invariants

The specification verifies these invariants:

| Invariant | Description |
|-----------|-------------|
| `MessageStateExclusive` | Each message is in exactly one state (pending, invisible, or deleted) |
| `HandleValidity` | Consumer handles match current valid handles |
| `DeliveryCountMonotonic` | Delivery counts are strictly increasing |
| `DeliveryCountPersistence` | Delivery counts persist across requeue operations |
| `NoMessageLoss` | Every message with data is pending or invisible |
| `HandleUniqueness` | Each delivery generates a unique handle |

## Liveness Properties

- `EventualRequeue` - Expired messages eventually return to pending (requires fairness)

## Installing TLA+ Tools

### macOS (Homebrew)

```bash
brew install tlaplus
```

### Manual Installation

1. Download from <https://github.com/tlaplus/tlaplus/releases>
2. Extract `tla2tools.jar` to a known location
3. Create an alias:

```bash
alias tlc='java -jar /path/to/tla2tools.jar'
```

## Running the Model Checker

### Basic Check

```bash
cd specs/tla
tlc RedisMailboxMC.tla -config RedisMailboxMC.cfg -workers auto
```

### Expected Output (No Errors)

```
TLC2 Version 2.18 of ...
Running breadth-first search Model-Checking with N workers...
...
Model checking completed. No error has been found.
  Fingerprint collision probability: calculated ...
```

### From Project Root

```bash
make tlaplus-check
```

## Configuration

The model uses small constants for exhaustive checking:

```
MaxMessages = 3        # Maximum messages to model
MaxDeliveries = 3      # Maximum deliveries per message
NumConsumers = 2       # Number of concurrent consumers
VisibilityTimeout = 2  # Timeout in abstract time units
```

Larger values exponentially increase state space size and checking time.

## Simulation Mode

For larger models, enable simulation mode by uncommenting in `RedisMailboxMC.cfg`:

```
SIMULATION
    NumSimulations = 1000
    TraceLength = 100
```

This runs probabilistic simulation instead of exhaustive checking.

## Correspondence to Implementation

| TLA+ Variable | Redis Data Structure |
|---------------|---------------------|
| `pending` | `{queue}:pending` LIST |
| `invisible` | `{queue}:invisible` ZSET |
| `data` | `{queue}:data` HASH |
| `handles` | `{queue}:meta` HASH (`:handle` field) |
| `deliveryCounts` | `{queue}:meta` HASH (`:count` field) |

| TLA+ Action | Lua Script |
|-------------|------------|
| `Receive` | `_LUA_RECEIVE` |
| `Acknowledge` | `_LUA_ACKNOWLEDGE` |
| `Nack` | `_LUA_NACK` |
| `Extend` | `_LUA_EXTEND` |
| `ReapOne` | `_LUA_REAP` |

## Troubleshooting

### "Invariant X is violated"

TLC found a counterexample. The error trace shows the sequence of actions
leading to the violation. Use this to identify and fix bugs in either the
specification or implementation.

### "Temporal property Y is violated"

A liveness property was violated. Check that fairness assumptions are satisfied
and that the property is correctly specified.

### State space explosion

If checking takes too long:

1. Reduce constant values in `RedisMailboxMC.cfg`
2. Enable simulation mode
3. Add symmetry sets if applicable

## References

- [TLA+ Home Page](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+](https://learntla.com/)
- [TLC Model Checker](https://lamport.azurewebsites.net/tla/tools.html)
- [specs/VERIFICATION.md](../VERIFICATION.md) - Full verification specification
