"""Print the CUDA devices visible to PyTorch."""

import torch


def main():
    device_count = torch.cuda.device_count()
    print(f"Available CUDA devices: {device_count}")
    for index in range(device_count):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")


if __name__ == "__main__":
    main()
