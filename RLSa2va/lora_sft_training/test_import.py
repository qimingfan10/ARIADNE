#!/usr/bin/env python3
import sys
print("Python version:", sys.version)
print("Starting imports...")

try:
    import torch
    print("✅ torch imported")
except Exception as e:
    print(f"❌ torch: {e}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ transformers imported")
except Exception as e:
    print(f"❌ transformers: {e}")

try:
    from peft import get_peft_model, LoraConfig
    print("✅ peft imported")
except Exception as e:
    print(f"❌ peft: {e}")

try:
    from combo_loss import ComboLoss
    print("✅ combo_loss imported")
except Exception as e:
    print(f"❌ combo_loss: {e}")

print("\n✅ All imports successful!")
print("\nNow testing training script...")

try:
    import train_lora_sft
    print("✅ train_lora_sft imported")
except Exception as e:
    print(f"❌ train_lora_sft import failed: {e}")
    import traceback
    traceback.print_exc()
