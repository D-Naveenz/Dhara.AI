# AGENTS.md

This workspace can use MindVault as optional local AI memory. Keep this file short: it is a router and quick reference, not the knowledge base.

## MindVault

- Use `$mindvault` / MindVault MCP to resolve local vault and workspace evidence.
- Workspace identity is stored in `mindvault.toml` as `workspace_id`.
- If MindVault tools are unavailable, continue from repo files only.
- Store durable lessons and cross-workspace principles in MindVault, not in this repository.

## Local Commands

- Build: `dotnet build Dhara.AI.slnx`
- Test: `dotnet test Dhara.AI.slnx`

## Local Guardrails

- Keep only repository-specific setup, commands, and hard safety rules here.
- Do not add local private paths or personal vault locations to this file.
