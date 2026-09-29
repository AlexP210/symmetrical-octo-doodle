import h5py
import sys

def visit(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}  shape={obj.shape} dtype={obj.dtype}")

with h5py.File(sys.argv[1], "r") as f:
    f.visititems(visit)