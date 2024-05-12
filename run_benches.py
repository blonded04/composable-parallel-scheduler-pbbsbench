#!/usr/bin/python3

import argparse
import os
import random
import string
import subprocess

parser = argparse.ArgumentParser(prog="run_benches")
parser.add_argument("--tbb-path", type=str, help="path to directory with tbb .so and /include")
parser.add_argument("--dir", type=str, help="put output into given dir instead of randomly generated")

parser.add_argument("--omp", action="store_true", help="set if want to run with OpenMP executor")
parser.add_argument("--tbb", action="store_true", help="set if want to run with oneTBB executor")
parser.add_argument("--eigen", action="store_true", help="set if want to run with eigen executor")

parser.add_argument("--numa", action="store_true", help="set if run on numa machine")
parser.add_argument("--nocheck", action="store_true", help="run without result validation")
parser.add_argument("--only", type=str, nargs='+', help="only run given test")
parser.add_argument("--small", action="store_true", help="set if want to test on smaller datasets")
parser.add_argument("--mode", type=str, help="run with specified mode")
parser.add_help = True

args = parser.parse_args()

if args.tbb and not args.tbb_path:
    print("if you want to run openTBB tests, --tbb-path is expected")
    exit(1)

print(args)

class Executor:
    def __init__(self, name, flag, modes):
        self.name = name
        self.flag = flag
        self.modes = modes


executors = []

if args.omp:
    omp_modes = [
        "OMP_STATIC",
        "OMP_DYNAMIC_MONOTONIC",
        "OMP_DYNAMIC_NONMONOTONIC",
        "OMP_GUIDED_MONOTONIC",
        "OMP_GUIDED_NONMONOTONIC",
    ]
    omp_executor = Executor(name="omp", flag=("OPENMP", 1), modes=omp_modes)

    executors.append(omp_executor)

if args.tbb:
    tbb_modes = [
        "TBB_SIMPLE",
        "TBB_AUTO",
        "TBB_AFFINITY",
        "TBB_CONST_AFFINITY",
    ]
    tbb_executor = Executor(name="tbb", flag=("TBB_PATH", args.tbb_path), modes=tbb_modes)

    executors.append(tbb_executor)

if args.eigen:
    eigen_modes = [
#        "EIGEN_SIMPLE",
#        "EIGEN_TIMESPAN",
#        "EIGEN_STATIC",
        "EIGEN_TIMESPAN_GRAINSIZE",
    ]
    eigen_executor = Executor(name="eigen", flag=("EIGEN", 1), modes=eigen_modes)

    executors.append(eigen_executor)

def random_str(length=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

if args.dir:
    target_dir = args.dir
else:
    target_dir = "bench_res_" + random_str()
    os.makedirs(target_dir)
print(f"target dir for this run is {target_dir}")

for executor in executors:
    flag, value = executor.flag
    exec_env = os.environ.copy()
    exec_env[flag] = str(value)

    mode_name = f"{executor.name.upper()}_MODE"

    if args.mode:
        modes = [args.mode]
    else:
        modes = executor.modes

    for mode in modes:
        run_env = exec_env.copy()
        run_env[mode_name] = mode
        print(f"now {executor.name} : {mode}")

        output_path = f"{target_dir}/{mode}.txt"
        output_file = open(output_path, "w")

        cmd = ["./runall", "-force", "-par"]

        if not args.numa:
            cmd.append("-nonuma")
        if args.nocheck:
            cmd.append("-nocheck")
        if args.only:
            cmd.append("-only")
            cmd.extend(args.only)
            cmd.append("-ext")
        if args.small:
            cmd.extend(["-small"])

        print("running command: ", cmd)
        process = subprocess.Popen(cmd, env=run_env, universal_newlines=True, stdout=output_file)
        rc = process.wait()
        if rc != 0:
            print(f"{mode_name} had bad returncode: {rc}")
