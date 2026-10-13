#!/usr/bin/env python3
"""Build a standalone bridge against installed SAPIEN, never replacing it.

Run with the exact DOMINO Python interpreter. The Ubuntu compiler packages
are only unpacked beneath external/native_camera_deps; no apt install/sudo.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tarfile

PYBIND_COMMIT = "cfe73560a252a4a0362f0bfe883c31c67770e266"
EXPECTED_ID = "__pybind11_internals_v4_gcc_libstdcpp_cxxabi1014__"
VERSION = "10.5.0-1ubuntu1~20.04"
PACKAGES = {
    "g++-10": "97107e42e3f0864811c809c8c3b7251af6c971395338f9ea8d58d916f724df95",
    "gcc-10": "cb987362235a84e00f13586f84fee48b1a7bd59c8207d86a238b25e7e0843271",
    "cpp-10": "5ca3be09f0ad34b3627ead2a4cdbd1dd8589e209a456c8dbbca7e1971e8bd871",
    "libstdc++-10-dev": "106ae470c9b34785f1b1ff06fa633d0899c8dd415979055aad8899b949ca2491",
    "libgcc-10-dev": "ebe5e967fca32ea5811946180f7822ee749857861eadaa29514e65857ce3b897",
    "gcc-10-base": "8fac06791057b1bba6178466e160bb3ec2a795297f10e88e34daf762572ed5c0",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(args, **kwargs):
    print("RUN", json.dumps(list(map(str, args))), flush=True)
    return subprocess.run(list(map(str, args)), check=True, text=True, **kwargs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--pybind-archive", type=Path, required=True)
    p.add_argument("--prepare-toolchain", action="store_true")
    args = p.parse_args()
    root = args.project.resolve()
    if not (root / "src/dynamicwam/runtime/domino_continuous").is_dir():
        raise RuntimeError("expected existing DynamicWAM project root")
    deps = root / "external/native_camera_deps"
    deps.mkdir(parents=True, exist_ok=True)
    gccroot = deps / "gcc10"
    gccroot.mkdir(exist_ok=True)
    packages = deps / "ubuntu-packages"
    packages.mkdir(exist_ok=True)
    if args.prepare_toolchain:
        for name, digest in PACKAGES.items():
            package = packages / f"{name}_{VERSION}_amd64.deb"
            if not package.exists():
                command(["apt-get", "download", f"{name}={VERSION}"], cwd=packages, timeout=300)
            if sha(package) != digest:
                raise RuntimeError(f"official apt metadata SHA256 mismatch: {name}")
            command(["dpkg-deb", "--extract", package, gccroot], timeout=120)
    compiler = gccroot / "usr/bin/g++-10"
    if not compiler.is_file():
        raise RuntimeError("GCC10 toolchain missing; use --prepare-toolchain")
    pybind = deps / f"pybind11-{PYBIND_COMMIT}"
    if not (pybind / "include/pybind11/smart_holder.h").exists():
        pybind.mkdir(exist_ok=True)
        with tarfile.open(args.pybind_archive) as archive:
            for member in archive.getmembers():
                resolved = (pybind / member.name).resolve()
                if pybind not in resolved.parents or not (member.isfile() or member.isdir()):
                    raise RuntimeError("unsafe dependency archive entry")
            archive.extractall(pybind)
    if importlib.metadata.version("sapien") != "3.0.0b1":
        raise RuntimeError("bridge is audited only for SAPIEN 3.0.0b1")
    spec = importlib.util.find_spec("sapien")
    sapien = Path(spec.origin).parent
    binding = next(sapien.glob("pysapien.cpython-310-*.so"))
    libdir = sapien.parent / "sapien.libs"
    library = libdir / "libsapien.so"
    binary = binding.read_bytes()
    if EXPECTED_ID.encode() + b"\0" not in binary:
        raise RuntimeError("installed pybind internals key differs; refusing forced ABI override")
    symbols = command(["nm", "-D", library], capture_output=True).stdout
    old_string_symbol = "_ZN6sapien15sapien_renderer27SapienRenderCameraComponent8getImageERKSs"
    if old_string_symbol not in symbols:
        raise RuntimeError("installed getImage std::string ABI is not verified old ABI=0")
    macros = command([compiler, "-dM", "-E", "-x", "c++", "-"], input="", capture_output=True).stdout
    if "#define __GXX_ABI_VERSION 1014" not in macros:
        raise RuntimeError("compiler does not naturally match installed cxxabi1014")
    include = sapien / "include"
    python_include = Path(sysconfig.get_path("include"))
    if not (python_include / "Python.h").exists():
        raise RuntimeError("Python C headers are missing")
    source = Path(__file__).with_name("sapien_color_reader.cpp")
    target = root / "src/dynamicwam/runtime/domino_continuous" / ("_sapien_color_reader" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = target.with_suffix(target.suffix + ".building")
    flags = [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", "-fvisibility=hidden",
             "-D_GLIBCXX_USE_CXX11_ABI=0", "-DPYBIND11_USE_SMART_HOLDER_AS_DEFAULT",
             "-DNDEBUG", "-I" + str(pybind / "include"), "-I" + str(python_include),
             "-I" + str(include), "-I" + str(include / "physx/include"), "-I/usr/include/eigen3",
             source, library, "-Wl,-rpath," + str(libdir), "-L/usr/lib/x86_64-linux-gnu",
             "-o", temporary]
    command(flags, timeout=180)
    check = r'''
import gc, importlib.util, json, sys
import sapien.pysapien as s
p=sys.argv[1]
spec=importlib.util.spec_from_file_location("_sapien_color_reader",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.pybind_internals_id=="__pybind11_internals_v4_gcc_libstdcpp_cxxabi1014__"
assert m.cpp_string_abi==0 and m.compiler_cxxabi==1014
camera=s.render.RenderCameraComponent(37,23)
for _ in range(20):
 try: m.read_color(camera)
 except RuntimeError as e: assert "not added to scene" in str(e), str(e)
 else: raise AssertionError("unattached camera unexpectedly succeeded")
try: m.read_color(object())
except (RuntimeError,TypeError): pass
else: raise AssertionError("wrong type accepted")
del camera;gc.collect()
print(json.dumps({"import":True,"registered_camera_cast":True,"native_exception_reacquires_gil":True,"wrong_type_rejected":True,"gpu_scene_created":False,"internals_id":m.pybind_internals_id}))
'''
    # importlib's extension suffix matching needs the final extension basename.
    temporary.replace(target)
    result = command([sys.executable, "-c", check, target], capture_output=True, timeout=60)
    print(result.stdout, flush=True)
    metadata = {"source_commit": PYBIND_COMMIT, "source_selection": "Latest first-parent commit in official archive/smart_holder before SAPIEN 3.0.0b1 PyPI release 2024-05-22; exact original wheel build commit unrecorded",
        "pybind_archive_sha256": sha(args.pybind_archive), "pybind_repository": "https://github.com/pybind/pybind11.git",
        "expected_internals_id":EXPECTED_ID,"abi_string_evidence":old_string_symbol,
        "compiler":str(compiler),"compiler_version":command([compiler,"--version"],capture_output=True).stdout,
        "compiler_packages":PACKAGES,"arguments":list(map(str,flags)),"python":sys.executable,
        "sapien_version":"3.0.0b1","sapien_binding_sha256":sha(binding),"sapien_library_sha256":sha(library),
        "source_sha256":sha(source),"artifact":str(target),"artifact_sha256":sha(target),
        "cpu_validation":json.loads(result.stdout.strip().splitlines()[-1]),
        "limits":"CPU import/cast/exception checks do not prove full ABI or renderer thread safety; actual GPU equivalence and physics overlap required before formal evaluation"}
    target.with_name("_sapien_color_reader.build.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("BRIDGE_READY",target,flush=True)

if __name__ == "__main__":
    main()
