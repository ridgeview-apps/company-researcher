# Analyst Review UI

A small TypeScript/React frontend for reviewing investigation findings
flagged for human review — approve, edit, reject, or request more research
against the findings and cited filing pages produced by the backend under
[`../backend/`](../backend/). See the root [`README.md`](../README.md) for
the full project.

This is the first slice: read/decide against findings the backend has
already flagged (`claim_type=interpretation` or `evidence_sufficient=false`).
It does not launch new investigations — that stays a `backend/` CLI command
for now.

## Setup

```bash
npm install
cp .env.example .env
```

`VITE_API_BASE_URL` in `.env` points at the FastAPI backend (defaults to
`http://localhost:8000`).

## Run

With the backend's API running (see the root README's "Run the API"
section), start the dev server:

```bash
npm run dev
```

Open <http://localhost:5173>.

## Quality checks

```bash
npm run lint   # oxlint
npm run build  # tsc -b (type check) + vite build
```
