"""miForge's MCP server (F12; BRD-MIFORGE-MCP-001).

Its own Deployment, its own process: it reaches miForge only through the REST API
(``client.ForgeClient``), so every rule in the REST layer - the approval gate above all -
applies to an agent's call. Nothing in this package imports the backend
(``tests/unit/test_mcp_stays_on_http.py``).
"""
