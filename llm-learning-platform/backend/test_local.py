import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.api.compute_routes import compute_attention, AttentionComputeRequest

async def main():
    req = AttentionComputeRequest(
        text="The cat sat on the mat",
        d_model=64,
        num_heads=4,
        num_layers=1,
        show_causal_mask=True
    )
    try:
        res = await compute_attention(req)
        import json
        json.dumps(res)
        print("Success! JSON serializable.")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
