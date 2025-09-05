"""
PyTorch 2.0 Installation Guide Script
This script provides platform-specific instructions to install and verify PyTorch 2.0.
It checks your Python version and optionally guides you to the appropriate commands.
"""

import sys
import platform
import subprocess

def check_python_version():
    print("\n1. Checking Python version...")
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 7):
        print("⚠️ PyTorch 2.0 requires Python >= 3.7. Please upgrade Python.")
    else:
        print("✅ Python version is compatible.")

def suggest_installation_commands():
    os_name = platform.system()
    machine = platform.machine()
    print(f"\n2. Detected OS: {os_name} | Architecture: {machine}")

    print("\n3. Suggested Installation Commands for PyTorch 2.0:\n")

    if os_name == "Darwin":  # macOS
        print("🔵 macOS Installation")
        print("Option 1: pip (CPU only):")
        print("  pip install torch torchvision torchaudio")
        print("\nOption 2: pip with MPS (Apple M1/M2 GPU support):")
        print("  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
        print("\nOption 3: conda (CPU only):")
        print("  conda install pytorch torchvision torchaudio -c pytorch")

    elif os_name == "Windows":
        print("🟢 Windows Installation")
        print("Option 1: pip (CPU only):")
        print("  pip install torch torchvision torchaudio")
        print("\nOption 2: pip with CUDA (GPU):")
        print("  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117")
        print("\nOption 3: conda (CPU only):")
        print("  conda install pytorch torchvision torchaudio -c pytorch")
        print("\nOption 4: conda with CUDA (GPU):")
        print("  conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch")

    elif os_name == "Linux":
        print("🟡 Linux Installation")
        print("Option 1: pip (CPU only):")
        print("  pip install torch torchvision torchaudio")
        print("\nOption 2: pip with CUDA (GPU):")
        print("  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117")
        print("\nOption 3: conda (CPU only):")
        print("  conda install pytorch torchvision torchaudio -c pytorch")
        print("\nOption 4: conda with CUDA (GPU):")
        print("  conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch")

    else:
        print("❗ Unsupported OS. Please refer to https://pytorch.org/get-started/locally/ for manual instructions.")

def verify_torch_installation():
    print("\n4. Verifying PyTorch Installation...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, "mps"):
            print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("❌ PyTorch is not installed.")
        print("Please install it using one of the suggested commands above.")

def main():
    print("==== PyTorch 2.0 Installation Guide ====")
    check_python_version()
    suggest_installation_commands()
    verify_torch_installation()

if __name__ == "__main__":
    main()
