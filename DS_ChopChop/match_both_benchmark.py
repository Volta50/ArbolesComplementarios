#!/usr/bin/env python3
"""
match_both_benchmark.py

Debug/benchmark script to compare:
 - Hybrid approach (greedy fast-path + backtracking fallback)
 - Backtracking-only approach

Measures total runtime, per-candidate times, number of backtracking calls, greedy successes, etc.

Usage examples:
    py match_both_benchmark.py --file arrays.txt --bt-timeout 3 --max-candidates 100
    py match_both_benchmark.py --a "[...]" --b "[...]" --max-candidates 50
"""
from typing import Optional, List, Tuple, Set, Dict
from collections import defaultdict
from functools import lru_cache
import argparse
import ast
import time
import concurrent.futures
import matplotlib.pyplot as plt

# --------------------
# Basic tree utilities (same as production script)
# --------------------
class Node:
    def __init__(self, val: int, left: 'Optional[Node]' = None, right: 'Optional[Node]' = None):
        self.val = val
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Node({self.val})"

def build_tree_from_list(vals: List[Optional[int]]) -> Optional[Node]:
    if not vals:
        return None
    nodes: List[Optional[Node]] = [None if v is None else Node(v) for v in vals]
    for i, node in enumerate(nodes):
        if node is None:
            continue
        li = 2*i + 1
        ri = 2*i + 2
        if li < len(nodes):
            node.left = nodes[li]
        if ri < len(nodes):
            node.right = nodes[ri]
    return nodes[0]

def final_leaves(root: Optional[Node], deletions: Set[str]) -> List[int]:
    res: List[int] = []
    def dfs(n: Optional[Node]):
        if n is None:
            return
        left_ok = (n.left is not None) and (f"{n.val}.L" not in deletions)
        right_ok = (n.right is not None) and (f"{n.val}.R" not in deletions)
        if not left_ok and not right_ok:
            res.append(n.val)
            return
        if left_ok:
            dfs(n.left)
        if right_ok:
            dfs(n.right)
    dfs(root)
    return res

def original_leaves(root: Optional[Node]) -> List[int]:
    return final_leaves(root, set())

# --------------------
# Candidate generation (validated) for smaller tree
# --------------------
def gen_pairs(node: Optional[Node], memo=None):
    if memo is None:
        memo = {}
    key = id(node) if node is not None else None
    if key in memo:
        return memo[key]
    results = set()
    if node is None:
        results.add(((), frozenset()))
        memo[key] = results
        return results
    if node.left is None and node.right is None:
        results.add(((node.val,), frozenset()))
        memo[key] = results
        return results
    left_pairs = gen_pairs(node.left, memo) if node.left is not None else { ((), frozenset()) }
    right_pairs = gen_pairs(node.right, memo) if node.right is not None else { ((), frozenset()) }
    # keep both
    for lt, ldel in left_pairs:
        for rt, rdel in right_pairs:
            tup = lt + rt
            deletions = set(ldel) | set(rdel)
            results.add((tup, frozenset(deletions)))
    # remove left
    if node.left is not None:
        for rt, rdel in right_pairs:
            tup = rt
            deletions = set(rdel) | {f"{node.val}.L"}
            results.add((tup, frozenset(deletions)))
    # remove right
    if node.right is not None:
        for lt, ldel in left_pairs:
            tup = lt
            deletions = set(ldel) | {f"{node.val}.R"}
            results.add((tup, frozenset(deletions)))
    # remove both
    delset = set()
    if node.left is not None: delset.add(f"{node.val}.L")
    if node.right is not None: delset.add(f"{node.val}.R")
    results.add(((node.val,), frozenset(delset)))
    memo[key] = results
    return results

def build_validated_map(root: Optional[Node]):
    raw_pairs = gen_pairs(root)
    mapk = defaultdict(lambda: defaultdict(set))
    for tup, ds in raw_pairs:
        simulated = tuple(final_leaves(root, set(ds)))
        if simulated == tup:
            mapk[len(tup)][tup].add(frozenset(ds))
    return mapk

# --------------------
# Backtracking matcher (same algorithm)
# --------------------
def find_deletions_to_match_exact(rootB: Node, target: Tuple[int, ...]) -> Optional[Set[str]]:
    nodes_by_id = {}
    def register(n: Optional[Node]):
        if n is None: return
        nodes_by_id[id(n)] = n
        register(n.left); register(n.right)
    register(rootB)
    n_target = len(target)
    @lru_cache(maxsize=None)
    def match(node_id:int, i:int):
        if node_id == 0:
            return [(i, frozenset())]
        node = nodes_by_id[node_id]
        results_local=[]
        if node.left is None and node.right is None:
            if i < n_target and target[i] == node.val:
                return [(i+1, frozenset())]
            return []
        left_exists = node.left is not None
        right_exists = node.right is not None
        left_opts = [('none', None)] if not left_exists else [('keep', id(node.left)), ('remove', None)]
        right_opts = [('none', None)] if not right_exists else [('keep', id(node.right)), ('remove', None)]
        for lopt, lref in left_opts:
            for ropt, rref in right_opts:
                if lopt=='remove' and ropt=='remove':
                    if i < n_target and target[i] == node.val:
                        ds=set()
                        if left_exists: ds.add(f"{node.val}.L")
                        if right_exists: ds.add(f"{node.val}.R")
                        results_local.append((i+1, frozenset(ds)))
                    continue
                left_matches=[(i, frozenset())]
                if lopt=='keep':
                    left_matches = match(lref, i)
                elif lopt=='remove':
                    left_matches = [(i, frozenset({f"{node.val}.L"}))]
                for (mid_idx, left_ds) in left_matches:
                    if ropt=='keep':
                        right_matches = match(rref, mid_idx)
                        for (end_idx, right_ds) in right_matches:
                            results_local.append((end_idx, frozenset(set(left_ds)|set(right_ds))))
                    elif ropt=='remove':
                        ds=set(left_ds); ds.add(f"{node.val}.R"); results_local.append((mid_idx, frozenset(ds)))
                    else:
                        results_local.append((mid_idx, left_ds))
        return results_local
    matches = match(id(rootB), 0)
    valid_ds = [ds for (j, ds) in matches if j == n_target]
    if not valid_ds:
        return None
    best = min(valid_ds, key=lambda s: (len(s), sorted(s)))
    return set(best)

# --------------------
# Greedy matcher (array-based) - same as before with safety checks
# --------------------
def build_index_helpers(arr):
    val2idx = {}
    parent = {}
    left = {}
    right = {}
    n = len(arr)
    for i,v in enumerate(arr):
        if v is None: continue
        val2idx[v] = i
    for i,v in enumerate(arr):
        if v is None: continue
        li = 2*i+1; ri = 2*i+2
        left[i] = li if li < n and arr[li] is not None else None
        right[i] = ri if ri < n and arr[ri] is not None else None
        if left[i] is not None: parent[left[i]] = i
        if right[i] is not None: parent[right[i]] = i
    return val2idx, parent, left, right

def compute_final_leaves_from_array(arr, deletions, left_map, right_map):
    res=[]
    def dfs(idx):
        if idx is None or arr[idx] is None: return
        v = arr[idx]
        lidx = left_map.get(idx)
        ridx = right_map.get(idx)
        left_ok = (lidx is not None) and (f"{v}.L" not in deletions)
        right_ok = (ridx is not None) and (f"{v}.R" not in deletions)
        if not left_ok and not right_ok:
            res.append(v); return
        if left_ok: dfs(lidx)
        if right_ok: dfs(ridx)
    if not arr or arr[0] is None: return []
    dfs(0); return res

def is_reachable(idx, arr, deletions, parent_map, left_map, right_map):
    cur = idx
    while cur in parent_map:
        p = parent_map[cur]
        pval = arr[p]
        if left_map.get(p) == cur:
            if f"{pval}.L" in deletions: return False
        else:
            if f"{pval}.R" in deletions: return False
        cur = p
    return True

def subtree_contains_any(target_idx, arr, fixed_indices_set, left_map, right_map):
    stack=[target_idx]
    while stack:
        cur = stack.pop()
        if cur in fixed_indices_set: return True
        l = left_map.get(cur); r = right_map.get(cur)
        if l is not None: stack.append(l)
        if r is not None: stack.append(r)
    return False

def greedy_match_using_array(arr, target):
    val2idx, parent_map, left_map, right_map = build_index_helpers(arr)
    for t in target:
        if t not in val2idx: return None
    deletions = set()
    fixed_vals = []
    fixed_indices_set = set()
    for t in target:
        idx = val2idx[t]
        if not is_reachable(idx, arr, deletions, parent_map, left_map, right_map):
            return None
        v = arr[idx]
        if left_map.get(idx) is not None: deletions.add(f"{v}.L")
        if right_map.get(idx) is not None: deletions.add(f"{v}.R")
        while True:
            curr_leaves = compute_final_leaves_from_array(arr, deletions, left_map, right_map)
            if v not in curr_leaves: return None
            if len(curr_leaves) <= len(fixed_vals): return None
            pos = curr_leaves.index(v)
            if pos < len(fixed_vals): return None
            if pos == len(fixed_vals):
                fixed_vals.append(v)
                fixed_indices_set.add(idx)
                break
            unwanted_val = curr_leaves[len(fixed_vals)]
            unwanted_idx = val2idx[unwanted_val]
            if unwanted_idx not in parent_map: return None
            p = parent_map[unwanted_idx]; pval = arr[p]
            if subtree_contains_any(unwanted_idx, arr, fixed_indices_set, left_map, right_map):
                return None
            if left_map.get(p) == unwanted_idx:
                deletions.add(f"{pval}.L")
            else:
                deletions.add(f"{pval}.R")
    final = compute_final_leaves_from_array(arr, deletions, left_map, right_map)
    return deletions if final == list(target) else None

# --------------------
# Backtracking worker + timeout wrapper
# --------------------
def backtracking_worker(arr: List[Optional[int]], target: Tuple[int, ...]):
    root = build_tree_from_list(arr)
    ds = find_deletions_to_match_exact(root, target)
    if ds is None:
        return None
    return sorted(list(ds))

def backtracking_with_timeout(arr: List[Optional[int]], target: Tuple[int, ...], timeout_sec: float):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(backtracking_worker, arr, tuple(target))
        try:
            res = fut.result(timeout=timeout_sec)
            if res is None: return None
            return set(res)
        except concurrent.futures.TimeoutError:
            try:
                fut.cancel()
            except Exception:
                pass
            return None

# --------------------
# Inorder pos precheck
# --------------------
def compute_inorder_positions(root: Optional[Node]) -> Dict[int,int]:
    pos = {}
    counter = {'c': 0}
    def dfs(n: Optional[Node]):
        if n is None: return
        dfs(n.left)
        pos[n.val] = counter['c']; counter['c'] += 1
        dfs(n.right)
    dfs(root); return pos

# --------------------
# Benchmark runs
# --------------------
def run_benchmark(arrA, arrB, bt_timeout=3.0, max_candidates=None, verbose=True):
    # Build trees and choose smaller
    A = build_tree_from_list(arrA)
    B = build_tree_from_list(arrB)
    leavesA = original_leaves(A); leavesB = original_leaves(B)
    if len(leavesA) <= len(leavesB):
        small_root, small_arr, large_root, large_arr = A, arrA, B, arrB
        small_name, large_name = "A","B"
    else:
        small_root, small_arr, large_root, large_arr = B, arrB, A, arrA
        small_name, large_name = "B","A"

    small_map = build_validated_map(small_root)
    # flatten candidates by descending k
    candidates = []
    for k in sorted(small_map.keys(), reverse=True):
        for t in sorted(small_map[k].keys()):
            candidates.append((k, t))
    if max_candidates:
        candidates = candidates[:max_candidates]

    # precompute inorder
    inorder_large = compute_inorder_positions(large_root)

    # Common prechecks function
    def prechecks_ok(target):
        # membership and inorder order
        for v in target:
            if v not in (x for x in large_arr if x is not None):
                return False
        try:
            inds = [inorder_large[v] for v in target]
            if any(inds[i] >= inds[i+1] for i in range(len(inds)-1)):
                return False
        except KeyError:
            return False
        return True

    # --- Hybrid run (greedy first, fallback to backtracking) ---
    hybrid_stats = {
        'start': time.perf_counter(),
        'candidates_tried': 0,
        'greedy_successes': 0,
        'backtracking_calls': 0,
        'backtracking_successes': 0,
        'time_per_candidate': [],
        'found': None,
        'found_ds_small': None,
        'found_ds_large': None,
        'found_target': None,
        'found_method': None
    }

    for k, target in candidates:
        hybrid_stats['candidates_tried'] += 1
        t0 = time.perf_counter()
        # cheap prechecks
        if not prechecks_ok(list(target)):
            hybrid_stats['time_per_candidate'].append(time.perf_counter()-t0)
            continue
        # greedy
        greedy_start = time.perf_counter()
        greedy_ds = greedy_match_using_array(large_arr, list(target))
        greedy_time = time.perf_counter() - greedy_start
        if greedy_ds is not None:
            # verify on node tree
            if final_leaves(large_root, greedy_ds) == list(target):
                hybrid_stats['greedy_successes'] += 1
                hybrid_stats['time_per_candidate'].append(time.perf_counter()-t0)
                hybrid_stats['found'] = True
                hybrid_stats['found_ds_small'] = set(next(iter(small_map[k][target])))
                hybrid_stats['found_ds_large'] = greedy_ds
                hybrid_stats['found_target'] = target
                hybrid_stats['found_method'] = 'greedy'
                break
            else:
                greedy_ds = None
        # fallback -> backtracking
        hybrid_stats['backtracking_calls'] += 1
        bt_start = time.perf_counter()
        back_ds = backtracking_with_timeout(large_arr, target, timeout_sec=bt_timeout)
        bt_time = time.perf_counter() - bt_start
        if back_ds is not None:
            hybrid_stats['backtracking_successes'] += 1
            hybrid_stats['found'] = True
            hybrid_stats['found_ds_small'] = set(next(iter(small_map[k][target])))
            hybrid_stats['found_ds_large'] = set(back_ds)
            hybrid_stats['found_target'] = target
            hybrid_stats['found_method'] = 'backtracking'
            hybrid_stats['time_per_candidate'].append(time.perf_counter()-t0)
            break
        hybrid_stats['time_per_candidate'].append(time.perf_counter()-t0)
    hybrid_stats['total_time'] = time.perf_counter() - hybrid_stats['start']

    # --- Backtracking-only run (no greedy) ---
    bt_stats = {
        'start': time.perf_counter(),
        'candidates_tried': 0,
        'backtracking_calls': 0,
        'backtracking_successes': 0,
        'time_per_candidate': [],
        'found': None,
        'found_ds_small': None,
        'found_ds_large': None,
        'found_target': None,
    }

    # We run over same candidates for fairness
    for k, target in candidates:
        bt_stats['candidates_tried'] += 1
        t0 = time.perf_counter()
        if not prechecks_ok(list(target)):
            bt_stats['time_per_candidate'].append(time.perf_counter()-t0)
            continue
        bt_stats['backtracking_calls'] += 1
        back_start = time.perf_counter()
        back_ds = backtracking_with_timeout(large_arr, target, timeout_sec=bt_timeout)
        back_time = time.perf_counter() - back_start
        if back_ds is not None:
            bt_stats['backtracking_successes'] += 1
            bt_stats['found'] = True
            bt_stats['found_ds_small'] = set(next(iter(small_map[k][target])))
            bt_stats['found_ds_large'] = set(back_ds)
            bt_stats['found_target'] = target
            bt_stats['time_per_candidate'].append(time.perf_counter()-t0)
            break
        bt_stats['time_per_candidate'].append(time.perf_counter()-t0)
    bt_stats['total_time'] = time.perf_counter() - bt_stats['start']

    # Print summary
    def print_summary():
        print("===== BENCHMARK SUMMARY =====")
        print("Candidates tested (limit):", len(candidates))
        print()
        print("HYBRID (greedy first, fallback to backtracking):")
        print("  total time: {:.3f}s".format(hybrid_stats['total_time']))
        print("  candidates tried: {}".format(hybrid_stats['candidates_tried']))
        print("  greedy successes: {}".format(hybrid_stats['greedy_successes']))
        print("  backtracking calls: {}".format(hybrid_stats['backtracking_calls']))
        print("  backtracking successes: {}".format(hybrid_stats['backtracking_successes']))
        if hybrid_stats['time_per_candidate']:
            print("  avg time/candidate: {:.4f}s".format(sum(hybrid_stats['time_per_candidate'])/len(hybrid_stats['time_per_candidate'])))
        if hybrid_stats['found']:
            print("  found target (method):", hybrid_stats['found_target'], hybrid_stats['found_method'])
            print("  deletions small:", hybrid_stats['found_ds_small'])
            print("  deletions large:", hybrid_stats['found_ds_large'])
        else:
            print("  no match found within tested candidates.")
        print()
        print("BACKTRACKING-ONLY:")
        print("  total time: {:.3f}s".format(bt_stats['total_time']))
        print("  candidates tried: {}".format(bt_stats['candidates_tried']))
        print("  backtracking calls: {}".format(bt_stats['backtracking_calls']))
        print("  backtracking successes: {}".format(bt_stats['backtracking_successes']))
        if bt_stats['time_per_candidate']:
            print("  avg time/candidate: {:.4f}s".format(sum(bt_stats['time_per_candidate'])/len(bt_stats['time_per_candidate'])))
        if bt_stats['found']:
            print("  found target:", bt_stats['found_target'])
            print("  deletions small:", bt_stats['found_ds_small'])
            print("  deletions large:", bt_stats['found_ds_large'])
        else:
            print("  no match found within tested candidates.")
        print("=============================")

    print_summary()

    return hybrid_stats, bt_stats

# --------------------
# CLI and main
# --------------------
def parse_array_string(s: str) -> List[Optional[int]]:
    s = s.strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            out=[]
            for x in val:
                if x is None: out.append(None)
                elif isinstance(x,int): out.append(x)
                elif isinstance(x,str) and x.lower() in ("none","null"):
                    out.append(None)
                else:
                    out.append(int(x))
            return out
    except Exception:
        pass
    parts=[p.strip() for p in s.split(",") if p.strip()!=""]
    out=[]
    for p in parts:
        if p.lower() in ("none","null"): out.append(None)
        else: out.append(int(p))
    return out

def read_arrays_from_file(path: str):
    lines=[]
    with open(path,'r', encoding='utf8') as f:
        for raw in f:
            s = raw.strip()
            if s: lines.append(s)
            if len(lines)>=2: break
    if len(lines)<2:
        raise ValueError("file must have two non-empty lines")
    return parse_array_string(lines[0]), parse_array_string(lines[1])

def main():
    p = argparse.ArgumentParser(description="Benchmark hybrid vs backtracking-only strategies.")
    p.add_argument('--file', '-f', help="File with two arrays (A then B).")
    p.add_argument('--a', help="Array A as list or comma-separated.")
    p.add_argument('--b', help="Array B as list or comma-separated.")
    p.add_argument('--bt-timeout', type=float, default=3.0, help="Backtracking per-candidate timeout in seconds.")
    p.add_argument('--max-candidates', type=int, default=200, help="Max candidates to test (for speed).")
    args = p.parse_args()

    if args.file:
        arrA, arrB = read_arrays_from_file(args.file)
    else:
        if not args.a or not args.b:
            print("Either --file or both --a and --b must be given.")
            return
        arrA = parse_array_string(args.a)
        arrB = parse_array_string(args.b)

    print("Running benchmark...")
    run_benchmark(arrA, arrB, bt_timeout=args.bt_timeout, max_candidates=args.max_candidates)

if __name__ == "__main__":
    # multiprocessing-safe entry
    main()
