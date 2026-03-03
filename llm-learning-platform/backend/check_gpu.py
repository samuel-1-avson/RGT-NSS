"""Verify GPU availability for PyTorch in Docker."""
import torch
import sys

print("=" * 60)
print("GPU Verification for LLM Learning Platform")
print("=" * 60)

# PyTorch version
print(f"\n📦 PyTorch Version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n🔥 CUDA Available: {cuda_available}")

if cuda_available:
    print(f"📊 CUDA Version: {torch.version.cuda}")
    print(f"🔢 cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"\n💻 GPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - Compute Capability: {props.major}.{props.minor}")
        print(f"    - Multi-Processor Count: {props.multi_processor_count}")
    
    # Test GPU computation
    print("\n🧪 Testing GPU computation...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✅ GPU computation successful! Result shape: {z.shape}")
        print(f"✅ Tensor device: {z.device}")
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        sys.exit(1)
    
    print("\n✅ GPU is ready for training!")
else:
    print("\n⚠️  WARNING: CUDA not available!")
    print("   The system will use CPU for training (much slower)")
    print("\n   Possible causes:")
    print("   - NVIDIA drivers not installed on host")
    print("   - Docker GPU runtime not configured")
    print("   - NVIDIA Container Toolkit not installed")
    sys.exit(1)
