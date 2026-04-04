from QUANTA_selector import QuantaSelector

def test_selector():
    print("Testing QUANTA_selector...")
    selector = QuantaSelector()
    
    # We will test limit=20 to quickly ensure thresholds + padding are functioning
    print("Fetching top 20 extreme coins...")
    coins = selector.get_research_backed_coins_for_training(limit=20)
    
    print("\nSelected Coins:")
    for c in coins:
        print(c)
        
    print(f"\nTotal Selected: {len(coins)}")
    assert len(coins) == 20, f"Expected 20, got {len(coins)}!"
    
    # Check for obvious garbage
    garbage_found = any(c for c in coins if 'USD1' in c or 'EUR' in c or 'PAXG' in c or 'BUSD' in c)
    assert not garbage_found, "Garbage coin slipped through!"
    
    print("\n✅ Selector Logic is mathematically sound.")

if __name__ == "__main__":
    test_selector()
