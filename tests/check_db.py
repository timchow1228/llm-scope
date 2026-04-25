import asyncio
import sys
sys.path.insert(0, '.')
from llm_scope.storage import init_db, get_calls

async def main():
    await init_db()
    rows = await get_calls(5)
    if not rows:
        print('NO ROWS FOUND IN DB')
        return
    print(f"Found {len(rows)} rows:")
    for r in rows:
        print(
            f"  id={r['id'][:8]}... "
            f"provider={r['provider']} "
            f"model={r['model']} "
            f"ttft={r['ttft_ms']:.0f}ms "
            f"total={r['total_ms']:.0f}ms "
            f"cost_usd={r['cost_usd']}"
        )

asyncio.run(main())
