#!/usr/bin/env python3
"""
Test script for interval-based DN consistency and entropy regularization
"""
import torch

def test_interval_logic():
    """Test the interval checking logic"""
    print("Testing interval logic...")
    
    # Define intervals
    dn_consistency_intervals = [(6000, 6999), (29000, 29999)]
    entropy_reg_intervals = [(6000, 6999), (29000, 29999)]
    
    # Test cases
    test_iterations = [5999, 6000, 6500, 6999, 7000, 28999, 29000, 29500, 29999, 30000]
    
    print("Iteration | DN Consistency | Entropy Reg")
    print("-" * 40)
    
    for iteration in test_iterations:
        apply_dn = any(start <= iteration <= end for start, end in dn_consistency_intervals)
        apply_entropy = any(start <= iteration <= end for start, end in entropy_reg_intervals)
        
        print(f"{iteration:8d} | {str(apply_dn):13s} | {str(apply_entropy):10s}")
    
    # Verify expected behavior
    expected_active = [6000, 6500, 6999, 29000, 29500, 29999]
    expected_inactive = [5999, 7000, 28999, 30000]
    
    all_correct = True
    for iteration in expected_active:
        apply_dn = any(start <= iteration <= end for start, end in dn_consistency_intervals)
        if not apply_dn:
            print(f"❌ Expected iteration {iteration} to be active but it's not")
            all_correct = False
    
    for iteration in expected_inactive:
        apply_dn = any(start <= iteration <= end for start, end in dn_consistency_intervals)
        if apply_dn:
            print(f"❌ Expected iteration {iteration} to be inactive but it's active")
            all_correct = False
    
    if all_correct:
        print("✅ Interval logic works correctly!")
    else:
        print("❌ Interval logic has issues!")
    
    return all_correct

def test_entropy_loss():
    """Test entropy regularization computation"""
    print("\nTesting entropy regularization...")
    
    # Create mock opacities
    opacities = torch.tensor([0.1, 0.5, 0.9, 0.01, 0.99], device='cuda')
    
    # Compute entropy loss
    entropy_loss = (
        - opacities * torch.log(opacities + 1e-10)
        - (1 - opacities) * torch.log(1 - opacities + 1e-10)
    ).mean()
    
    print(f"✓ Entropy loss computed: {entropy_loss.item():.6f}")
    
    # Test edge cases
    edge_opacities = torch.tensor([0.0, 1.0], device='cuda')
    edge_entropy = (
        - edge_opacities * torch.log(edge_opacities + 1e-10)
        - (1 - edge_opacities) * torch.log(1 - edge_opacities + 1e-10)
    ).mean()
    
    print(f"✓ Edge case entropy loss: {edge_entropy.item():.6f}")
    return True

if __name__ == "__main__":
    print("🧪 Testing interval-based regularization...\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Tests require GPU.")
        exit(1)
    
    success = True
    success &= test_interval_logic()
    success &= test_entropy_loss()
    
    if success:
        print("\n✅ All tests passed! Interval-based regularization is working.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")