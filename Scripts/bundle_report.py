"""Size report for a frozen BabelBrain bundle, plus a DLL check on Windows.

    python Scripts/bundle_report.py BabelBrain/dist/version/BabelBrain.app
    python Scripts/bundle_report.py BabelBrain/dist/version/BabelBrain [--strict]

Prints the total size, the largest files and the largest top-level folders,
so the effect of BabelBrain.spec pruning can be read from the CI log.

On Windows it also parses every .dll/.pyd with pefile and lists load-time
imports that neither the bundle nor System32 provides. Imports of the CUDA math
libraries pruned by BabelBrain.spec are expected (CuPy imports the wrappers
that need them lazily) and are listed separately. Anything else is a library
the spec dropped that some bundled module still links against; --strict turns
that into a non-zero exit.
"""
import argparse
import os
import sys
from collections import defaultdict

# Keep in step with CUDA_MATH_LIBS in BabelBrain/BabelBrain.spec
CUDA_MATH_LIBS = ("cublas", "cufft", "curand", "cusolver", "cusparse",
                  "cutensor", "nccl", "cudnn")
# Provided by the system at run time, never bundled
SYSTEM_PROVIDED = ("nvcuda.dll",)


def mb(n):
    return f"{n / 1048576:9.1f} MB"


def walk_files(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            path = os.path.join(dirpath, f)
            if not os.path.islink(path):
                yield path, os.path.getsize(path)


def size_report(root, top):
    files = list(walk_files(root))
    total = sum(s for _, s in files)
    print(f"Bundle: {root}\nTotal: {mb(total)} in {len(files)} files\n")

    print(f"Largest {top} files:")
    for path, size in sorted(files, key=lambda x: -x[1])[:top]:
        print(f"  {mb(size)}  {os.path.relpath(path, root)}")

    # Group by the first folder below the folder that holds the libraries
    lib_root = root
    for candidate in ("Contents/Frameworks", "_internal"):
        if os.path.isdir(os.path.join(root, candidate)):
            lib_root = os.path.join(root, candidate)
    groups = defaultdict(int)
    for path, size in files:
        rel = os.path.relpath(path, lib_root)
        if rel.startswith(os.pardir + os.sep):
            head = "(outside " + os.path.basename(lib_root) + ")"
        else:
            head = rel.split(os.sep)[0] if os.sep in rel else "(loose library files)"
        groups[head] += size
    print(f"\nLargest {top} folders under {os.path.relpath(lib_root, root)}:")
    for name, size in sorted(groups.items(), key=lambda x: -x[1])[:top]:
        print(f"  {mb(size)}  {name}")


def windows_dll_check(root):
    import pefile

    bundled = {}
    for path, _ in walk_files(root):
        if path.lower().endswith((".dll", ".pyd", ".exe")):
            bundled.setdefault(os.path.basename(path).lower(), path)
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    system = {f.lower() for f in os.listdir(system32)} if os.path.isdir(system32) else set()

    missing = defaultdict(list)
    for name, path in bundled.items():
        try:
            pe = pefile.PE(path, fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        except pefile.PEFormatError:
            continue
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll = entry.dll.decode(errors="replace").lower()
            if (dll in bundled or dll in system or dll in SYSTEM_PROVIDED
                    or dll.startswith(("api-ms-", "ext-ms-"))):
                continue
            missing[dll].append(os.path.relpath(path, root))
        pe.close()

    expected = {d: v for d, v in missing.items() if d.startswith(CUDA_MATH_LIBS)}
    unexpected = {d: v for d, v in missing.items() if d not in expected}

    print("\nPruned CUDA math libraries and the (lazily imported) modules that link them:")
    for dll, users in sorted(expected.items()):
        print(f"  {dll}: {', '.join(sorted(users))}")
    if not expected:
        print("  (none)")

    print("\nUnresolved load-time DLL imports:")
    for dll, users in sorted(unexpected.items()):
        print(f"  {dll}: {', '.join(sorted(users))}")
    if not unexpected:
        print("  (none)")
    return unexpected


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="dist folder (Windows) or .app bundle (macOS)")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on unresolved DLL imports (Windows)")
    args = ap.parse_args()

    size_report(args.bundle, args.top)
    if sys.platform == "win32":
        unexpected = windows_dll_check(args.bundle)
        if unexpected and args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
