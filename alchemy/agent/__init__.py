"""Visuomotor agent — screenshot → UI-TARS-72B → action → xdotool.

The agent loop runs inside Alchemy (CPU side). It captures screenshots from
the shadow desktop, sends them to UI-TARS-72B for analysis, and executes
the resulting actions via xdotool in WSL2.

For APPROVE-tier actions, the agent pauses and requests approval from
NEO-TX (the user-facing layer) before executing.
"""
