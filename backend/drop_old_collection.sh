#!/usr/bin/env bash
# Drop the retired bcit_docs collection (one-off cleanup after the 2026-06 rebuild).
set -e
cd "$(dirname "$0")"

# Reuse a proxy already listening on 5433, otherwise start one
if ! (exec 3<>/dev/tcp/127.0.0.1/5433) 2>/dev/null; then
  ~/cloud-sql-proxy --port 5433 wine-agent-jh-2026:us-west1:bcit-rag-pg &
  sleep 4
fi

export PG_CONNECTION=$(grep '^PG_CONNECTION=' .env | cut -d= -f2- | sed 's/:5432/:5433/')
.venv/bin/python - <<'PYEOF'
import os
from sqlalchemy import create_engine, text
e = create_engine(os.environ['PG_CONNECTION'])
with e.connect() as c:
    c = c.execution_options(isolation_level="AUTOCOMMIT")
    n = c.execute(text("DELETE FROM langchain_pg_embedding WHERE collection_id=(SELECT uuid FROM langchain_pg_collection WHERE name='bcit_docs')")).rowcount
    c.execute(text("DELETE FROM langchain_pg_collection WHERE name='bcit_docs'"))
    c.execute(text("VACUUM ANALYZE langchain_pg_embedding"))
    print("deleted old chunks:", n)
    rows = c.execute(text("SELECT c.name, count(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id=c.uuid GROUP BY c.name")).fetchall()
    print("remaining:", rows)
PYEOF
