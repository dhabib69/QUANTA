import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:65087'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:65087'

from QUANTA_selector import QuantaSelector
import time

def validate_200_coins_fetch():
    print("Testing QUANTA_selector robust 200 coins fetching...")
    selector = QuantaSelector()
    
    start_time = time.time()
    
    print("Fetching top 200 extreme coins... (This will test multi-threading limit handling & rate limits)")
    coins = selector.get_research_backed_coins_for_training(limit=200)
    
    elapsed = time.time() - start_time
    
    print(f"\nFinal Selected Coins ({len(coins)}/200):")
    for i, c in enumerate(coins):
        print(f"{i+1:03d}: {c}")
        
    print(f"\nTime taken: {elapsed:.2f} seconds")
    
    if len(coins) < 200:
        print(f"⚠️ WARNING: Rate limits or filters kicked in. Only fetched {len(coins)} coins instead of 200.")
    else:
        print("✅ SUCCESS: Successfully fetched 200 coins without getting completely banned by proxy/rate limit.")
        
    # Check for obvious garbage
    garbage_found = any(c for c in coins if c in ['USD1USDT', 'EURUSDT', 'PAXGUSDT', 'BUSDUSDT'])
    assert not garbage_found, "Garbage coin slipped through!"

if __name__ == "__main__":
    validate_200_coins_fetch()
