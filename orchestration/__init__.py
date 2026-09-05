"""Run_Plan_Spec §1.3 orchestration: manifest + mkdir-claim primitives.

Kept as a library (not scripts) so run_worker.py, manifest_incomplete.py,
and the tests all share the same code path — no drift between the
worker's claim logic and the status dashboard's read-only inspection.
"""
