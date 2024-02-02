#!/usr/bin/python3

import fileinput
import json
import typing

def extract_bench_name(line: str) -> typing.Optional[str]:
    prefix = "cd benchmarks/"
    if not line.startswith(prefix):
        return None

    line = line[len(prefix):]
    line = line[:line.find(" ")]
    return line

def extract_measures(line: str) -> typing.Tuple[typing.List[float], str]:
    l_idx = line.find("'") + 1
    r_idx = None
    measures = []

    while l_idx != 0:
        line = line[l_idx:]
        r_idx = line.find("'")
        assert r_idx != -1

        measures.append(float(line[:r_idx]))
        line = line[r_idx + 1:]
        l_idx = line.find("'") + 1

    return (measures, line)

def extract_geomean(line: str) -> float:
    idx = line.find(" = ") + 3
    return float(line[idx:])

def extract_bench_res(line: str) -> typing.Optional[typing.Tuple[str, typing.List[float], float]]:
    if line.find("geomean = ") == -1:
        return None

    dataset = line[:line.find(" ")]
    measures_str = line.split(":")[2]
    measures, rest = extract_measures(measures_str)
    geomean = extract_geomean(rest)
    return (dataset, measures, geomean)

def extract_total_res(line: str) -> typing.Optional[typing.Tuple[str, float, float]]:
    if line.find("geomeans =") == -1:
        return None

    name = line[:line.find(" ")]
    mins_str = "geomean of mins = "
    total_str = "geomean of geomeans = "

    mins_l_idx = line.find(mins_str) + len(mins_str)
    line = line[mins_l_idx:]
    mins = float(line[:line.find(",")])

    total_l_idx = line.find(total_str) + len(total_str)
    total = float(line[total_l_idx:])

    return (name, mins, total)


current_bench = None
result = {}

for line in fileinput.input():
    # bench_name = extract_bench_name(line)
    bench_res = extract_total_res(line)

    if bench_res:
        dataset, g_of_mins, g_of_g = bench_res

        result[dataset] = {
            "mins": g_of_mins,
            "total": g_of_g,
        }

print(json.dumps(result))
