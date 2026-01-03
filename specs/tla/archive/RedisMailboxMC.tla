---------------------------- MODULE RedisMailboxMC ----------------------------
(* Model Checking Configuration Module for RedisMailbox
 *
 * This module extends RedisMailbox with small constants suitable for
 * exhaustive model checking with TLC.
 *
 * Usage:
 *   tlc RedisMailboxMC.tla -config RedisMailboxMC.cfg -workers auto
 *)

EXTENDS RedisMailbox

=============================================================================
