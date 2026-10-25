# Benchmark integrations

This directory contains local specifications and integration notes, not copies
of upstream projects.

| Integration | Pinned implementation | Data/output |
|---|---|---|
| VitaBench | `vendor/vitabench/` | vendor fixtures; generated output in `tau2bench-data/artifacts/benchmarks/vita/` |
| Persona Policies | `vendor/persona_policies/` | generated output in `tau2bench-data/artifacts/benchmarks/persona_policies/` |
| User Behavioral Divergence | `vendor/user_behavioral_divergence/` | generated output in the external data workspace |
| USI reference | `src/cresta/evaluation/usi/` | `datasets/usi/` |

Pinned revisions are recorded in `PROVENANCE.md`. Vendor projects retain their
own dependency files and must not import experiment launchers from this repo.
