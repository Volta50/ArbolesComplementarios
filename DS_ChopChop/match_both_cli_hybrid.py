#!/usr/bin/env python3
"""
match_both_cli_greedy_with_timeout.py

Updated script that integrates:
 - greedy fast-path (array-based)
 - validated enumeration on the smaller tree
 - backtracking fallback executed with a timeout using ProcessPoolExecutor
 - cheap pre-check (inorder positions) to prune impossible candidates
 - CLI flags: --no-greedy, --bt-timeout, --save-dir

Save and run:
    py match_both_cli_greedy_with_timeout.py --file arrays.txt --save-dir out_plots
"""

from typing import Optional, List, Tuple, Set, Dict
from collections import defaultdict
from functools import lru_cache
import argparse
import ast
import os
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import time

# --------------------
# Node + builder
# --------------------
class Node:
    def __init__(self, val: int, left: 'Optional[Node]' = None, right: 'Optional[Node]' = None):
        self.val = val
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Node({self.val})"

def build_tree_from_list(vals: List[Optional[int]]) -> Optional[Node]:
    """Build tree from level-order list where None = missing node."""
    if not vals:
        return None
    nodes: List[Optional[Node]] = [None if v is None else Node(v) for v in vals]
    for i, node in enumerate(nodes):
        if node is None:
            continue
        li = 2 * i + 1
        ri = 2 * i + 2
        if li < len(nodes):
            node.left = nodes[li]
        if ri < len(nodes):
            node.right = nodes[ri]
    return nodes[0]

# --------------------
# CLI parsing helpers
# --------------------
def parse_array_string(s: str) -> List[Optional[int]]:
    s = s.strip()
    if not s:
        return []
    # Try Python literal list first
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            out = []
            for x in val:
                if x is None:
                    out.append(None)
                elif isinstance(x, int):
                    out.append(x)
                elif isinstance(x, str) and x.lower() in ("none", "null"):
                    out.append(None)
                else:
                    out.append(int(x))
            return out
    except Exception:
        pass
    # Fallback comma-separated
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        if p.lower() in ("none", "null"):
            out.append(None)
        else:
            out.append(int(p))
    return out

def read_arrays_from_file(path: str) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    lines = []
    with open(path, 'r', encoding='utf8') as f:
        for raw in f:
            s = raw.strip()
            if s:
                lines.append(s)
            if len(lines) >= 2:
                break
    if len(lines) < 2:
        raise ValueError("Input file must contain at least two non-empty lines (Tree A then Tree B).")
    return parse_array_string(lines[0]), parse_array_string(lines[1])

# --------------------
# Leaves, pretty printing, final leaves after deletions
# --------------------
def original_leaves(root: Optional[Node]) -> List[int]:
    res: List[int] = []
    def dfs(n: Optional[Node]):
        if n is None:
            return
        if n.left is None and n.right is None:
            res.append(n.val)
            return
        dfs(n.left)
        dfs(n.right)
    dfs(root)
    return res

def pretty_print_tree(root: Optional[Node], prefix: str = "") -> None:
    if root is None:
        print(prefix + "<empty>")
        return
    print(prefix + f"{root.val}")
    if root.left is None and root.right is None:
        return
    if root.left is not None:
        pretty_print_tree(root.left, prefix + "  L-")
    else:
        print(prefix + "  L-<none>")
    if root.right is not None:
        pretty_print_tree(root.right, prefix + "  R-")
    else:
        print(prefix + "  R-<none>")

def pretty_print_with_deletions(n: Optional[Node], deletions: Set[str], prefix: str = "") -> None:
    if n is None:
        print(prefix + "<empty>")
        return
    print(prefix + f"{n.val}")
    if n.left is None:
        print(prefix + "  L-<none>")
    else:
        if f"{n.val}.L" in deletions:
            print(prefix + f"  L-<deleted edge to {n.left.val}>")
        else:
            pretty_print_with_deletions(n.left, deletions, prefix + "  L-")
    if n.right is None:
        print(prefix + "  R-<none>")
    else:
        if f"{n.val}.R" in deletions:
            print(prefix + f"  R-<deleted edge to {n.right.val}>")
        else:
            pretty_print_with_deletions(n.right, deletions, prefix + "  R-")

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

# --------------------
# Symbolic generation + validation (for smaller tree)
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

    # remove both -> node leaf
    delset = set()
    if node.left is not None:
        delset.add(f"{node.val}.L")
    if node.right is not None:
        delset.add(f"{node.val}.R")
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
# Backtracking matcher on Node-tree (exact match)
# --------------------
def find_deletions_to_match_exact(rootB: Node, target: Tuple[int, ...]) -> Optional[Set[str]]:
    nodes_by_id = {}
    def register(n: Optional[Node]):
        if n is None:
            return
        nodes_by_id[id(n)] = n
        register(n.left); register(n.right)
    register(rootB)
    n_target = len(target)

    @lru_cache(maxsize=None)
    def match(node_id: int, i: int):
        if node_id == 0:
            return [(i, frozenset())]
        node = nodes_by_id[node_id]
        results_local = []
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
                if lopt == 'remove' and ropt == 'remove':
                    if i < n_target and target[i] == node.val:
                        ds = set()
                        if left_exists: ds.add(f"{node.val}.L")
                        if right_exists: ds.add(f"{node.val}.R")
                        results_local.append((i+1, frozenset(ds)))
                    continue
                left_matches = [(i, frozenset())]
                if lopt == 'keep':
                    left_matches = match(lref, i)
                elif lopt == 'remove':
                    left_matches = [(i, frozenset({f"{node.val}.L"}))]
                for (mid_idx, left_ds) in left_matches:
                    if ropt == 'keep':
                        right_matches = match(rref, mid_idx)
                        for (end_idx, right_ds) in right_matches:
                            results_local.append((end_idx, frozenset(set(left_ds) | set(right_ds))))
                    elif ropt == 'remove':
                        ds = set(left_ds)
                        ds.add(f"{node.val}.R")
                        results_local.append((mid_idx, frozenset(ds)))
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
# Greedy matcher (array based) - fast-path
# --------------------
def build_index_helpers(arr):
    val2idx = {}
    parent = {}
    left = {}
    right = {}
    n = len(arr)
    for i, v in enumerate(arr):
        if v is None:
            continue
        val2idx[v] = i
    for i, v in enumerate(arr):
        if v is None:
            continue
        li = 2*i + 1; ri = 2*i + 2
        left[i] = li if li < n and arr[li] is not None else None
        right[i] = ri if ri < n and arr[ri] is not None else None
        if left[i] is not None:
            parent[left[i]] = i
        if right[i] is not None:
            parent[right[i]] = i
    return val2idx, parent, left, right

def compute_final_leaves_from_array(arr, deletions, left_map, right_map):
    res = []
    def dfs(idx):
        if idx is None or arr[idx] is None:
            return
        v = arr[idx]
        lidx = left_map.get(idx)
        ridx = right_map.get(idx)
        left_ok = (lidx is not None) and (f"{v}.L" not in deletions)
        right_ok = (ridx is not None) and (f"{v}.R" not in deletions)
        if not left_ok and not right_ok:
            res.append(v)
            return
        if left_ok:
            dfs(lidx)
        if right_ok:
            dfs(ridx)
    if not arr or arr[0] is None:
        return []
    dfs(0)
    return res

def is_reachable(idx, arr, deletions, parent_map, left_map, right_map):
    cur = idx
    while cur in parent_map:
        p = parent_map[cur]
        pval = arr[p]
        if left_map.get(p) == cur:
            if f"{pval}.L" in deletions:
                return False
        else:
            if f"{pval}.R" in deletions:
                return False
        cur = p
    return True

def subtree_contains_any(target_idx, arr, fixed_indices_set, left_map, right_map):
    stack = [target_idx]
    while stack:
        cur = stack.pop()
        if cur in fixed_indices_set:
            return True
        l = left_map.get(cur)
        r = right_map.get(cur)
        if l is not None:
            stack.append(l)
        if r is not None:
            stack.append(r)
    return False

def greedy_match_using_array(arr, target):
    val2idx, parent_map, left_map, right_map = build_index_helpers(arr)
    # quick membership check
    for t in target:
        if t not in val2idx:
            return None

    deletions = set()
    fixed_vals = []
    fixed_indices_set = set()

    for t in target:
        idx = val2idx[t]

        # must be reachable from root under current deletions
        if not is_reachable(idx, arr, deletions, parent_map, left_map, right_map):
            return None

        v = arr[idx]
        # make idx a leaf by chopping its children (if any)
        if left_map.get(idx) is not None:
            deletions.add(f"{v}.L")
        if right_map.get(idx) is not None:
            deletions.add(f"{v}.R")

        # now remove earlier (unfixed) leaves until v becomes the next unfixed leaf
        while True:
            curr_leaves = compute_final_leaves_from_array(arr, deletions, left_map, right_map)

            # If v disappeared (disconnected), abort
            if v not in curr_leaves:
                return None

            # If there are not enough leaves left to have an unfixed one at position len(fixed_vals), abort
            if len(curr_leaves) <= len(fixed_vals):
                return None

            pos = curr_leaves.index(v)

            # If v is already before the fixed prefix, inconsistent -> abort
            if pos < len(fixed_vals):
                return None

            # If v is at the exact next position, fix it and continue
            if pos == len(fixed_vals):
                fixed_vals.append(v)
                fixed_indices_set.add(idx)
                break

            # Otherwise, there is an unwanted leaf before v that we should remove
            unwanted_val = curr_leaves[len(fixed_vals)]
            unwanted_idx = val2idx[unwanted_val]

            # Can't remove the root (no parent)
            if unwanted_idx not in parent_map:
                return None

            p = parent_map[unwanted_idx]
            pval = arr[p]

            # avoid cutting a subtree that contains any already-fixed index
            if subtree_contains_any(unwanted_idx, arr, fixed_indices_set, left_map, right_map):
                return None

            # delete the appropriate child-edge
            if left_map.get(p) == unwanted_idx:
                deletions.add(f"{pval}.L")
            else:
                deletions.add(f"{pval}.R")

            # loop will recompute curr_leaves and try again

    # After processing all target items, final verification
    final = compute_final_leaves_from_array(arr, deletions, left_map, right_map)
    return deletions if final == list(target) else None

# --------------------
# Pre-check helpers (inorder positions)
# --------------------
def compute_inorder_positions(root: Optional[Node]) -> Dict[int, int]:
    pos = {}
    counter = {'c': 0}
    def dfs(n: Optional[Node]):
        if n is None:
            return
        dfs(n.left)
        pos[n.val] = counter['c']
        counter['c'] += 1
        dfs(n.right)
    dfs(root)
    return pos

# --------------------
# Backtracking worker + timeout wrapper (for ProcessPoolExecutor)
# --------------------
def backtracking_worker(arr: List[Optional[int]], target: Tuple[int, ...]):
    """
    Worker executed in a separate process. Rebuilds the Node tree from arr
    and calls find_deletions_to_match_exact to obtain deletions (or None).
    Returns a sorted list of deletions or None.
    """
    root = build_tree_from_list(arr)
    ds = find_deletions_to_match_exact(root, target)
    if ds is None:
        return None
    return sorted(list(ds))

def backtracking_with_timeout(arr: List[Optional[int]], target: Tuple[int, ...], timeout_sec: float):
    """
    Run backtracking_worker in a separate process and wait up to timeout_sec.
    Returns a set of deletions (strings) if found, otherwise None.
    """
    # Use ProcessPoolExecutor so we can timeout and kill worker on Windows/Linux
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(backtracking_worker, arr, tuple(target))
        try:
            res = fut.result(timeout=timeout_sec)
            if res is None:
                return None
            return set(res)
        except concurrent.futures.TimeoutError:
            try:
                fut.cancel()
            except Exception:
                pass
            return None

# --------------------
# Plot helpers
# --------------------
def compute_positions(root: Optional[Node]):
    positions = {}
    counter = {'x': 0}
    def inorder(n: Optional[Node], depth=0):
        if n is None:
            return
        inorder(n.left, depth+1)
        positions[n] = (counter['x'], -depth)
        counter['x'] += 1
        inorder(n.right, depth+1)
    inorder(root, 0)
    return positions

def get_reachable_nodes(root: Optional[Node], deletions: Set[str]):
    reachable = set()
    def dfs(n: Optional[Node]):
        if n is None:
            return
        reachable.add(n)
        if n.left is not None and f"{n.val}.L" not in deletions:
            dfs(n.left)
        if n.right is not None and f"{n.val}.R" not in deletions:
            dfs(n.right)
    dfs(root)
    return reachable

def draw_tree(root: Optional[Node], deletions: Set[str]=None, title: str=None, save_path: Optional[str]=None):
    if deletions is None:
        deletions = set()
    positions = compute_positions(root)
    reachable = get_reachable_nodes(root, deletions)
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.set_title(title if title is not None else "Tree (after deletions)")
    ax.axis('off')
    for parent in list(reachable):
        if parent not in positions:
            continue
        px, py = positions[parent]
        if parent.left is not None and f"{parent.val}.L" not in deletions and parent.left in reachable:
            cx, cy = positions[parent.left]
            ax.plot([px, cx], [py, cy], linewidth=1)
        if parent.right is not None and f"{parent.val}.R" not in deletions and parent.right in reachable:
            cx, cy = positions[parent.right]
            ax.plot([px, cx], [py, cy], linewidth=1)
    for n, (x, y) in positions.items():
        if n in reachable:
            ax.scatter([x], [y], s=40)
            ax.text(x + 0.02, y, str(n.val), va='center', ha='left', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

# --------------------
# Main flow (greedy fast-path + prechecks + backtracking with timeout)
# --------------------
def main():
    p = argparse.ArgumentParser(description="Match leaf-arrays between two trees by chopping edges (greedy fast-path + backtracking with timeout).")
    p.add_argument('--file', '-f', help="Path to text file with two non-empty lines (arrays for A then B).")
    p.add_argument('--a', help="Array for tree A (python list or comma-separated), e.g. \"[1,2,3,None]\" or \"1,2,3,None\"")
    p.add_argument('--b', help="Array for tree B (python list or comma-separated).")
    p.add_argument('--save-dir', help="Optional directory to save visualization PNGs (will be created).")
    p.add_argument('--no-greedy', action='store_true', help="Disable greedy fast-path; force backtracking only.")
    p.add_argument('--bt-timeout', type=float, default=3.0, help="Timeout in seconds for backtracking per candidate (default 3.0).")
    args = p.parse_args()

    if args.file:
        arrA, arrB = read_arrays_from_file(args.file)
    else:
        if not args.a or not args.b:
            print("Either --file or both --a and --b must be provided. See --help.")
            return
        arrA = parse_array_string(args.a)
        arrB = parse_array_string(args.b)

    # Build Node trees
    A = build_tree_from_list(arrA)
    B = build_tree_from_list(arrB)

    leavesA = original_leaves(A)
    leavesB = original_leaves(B)
    print("Original Tree A leaves (L->R):", leavesA)
    print("Original Tree B leaves (L->R):", leavesB)
    print()

    # Choose smaller by leaves
    if len(leavesA) <= len(leavesB):
        small_root, small_name, small_arr = A, "A", arrA
        large_root, large_name, large_arr = B, "B", arrB
    else:
        small_root, small_name, small_arr = B, "B", arrB
        large_root, large_name, large_arr = A, "A", arrA

    print(f"Enumerating validated reachable leaf-arrays on smaller tree (Tree {small_name})...")
    small_map = build_validated_map(small_root)

    # prepare precomputed info for large tree
    inorder_positions_large = compute_inorder_positions(large_root)
    # backtracking cache: map (tuple(arr), target_tuple) -> Optional[set(deletions)]
    backtrack_cache = {}

    # search candidates descending k
    start_k = max(small_map.keys()) if small_map else 0
    found = False
    chosen_k = None
    chosen_array = None
    chosen_del_small = None
    chosen_del_large = None
    reason = None

    for k in range(start_k, 0, -1):
        candidates = sorted(small_map.get(k, {}).keys())
        if not candidates:
            continue
        for arr_tuple in candidates:
            # quick membership check: all values exist in large array
            if any(v not in (x for x in large_arr if x is not None) for v in arr_tuple):
                # impossible: some value not present in large tree
                continue

            # quick inorder pre-check: positions must be increasing (left-to-right)
            try:
                indices = [inorder_positions_large[v] for v in arr_tuple]
                if any(indices[i] >= indices[i+1] for i in range(len(indices)-1)):
                    # impossible due to relative order in tree (no reordering by chops)
                    continue
            except KeyError:
                # some value missing in inorder mapping -> skip (shouldn't happen after membership check)
                continue

            # Try greedy unless disabled
            greedy_ds = None
            if not args.no_greedy:
                greedy_ds = greedy_match_using_array(large_arr, list(arr_tuple))
                if greedy_ds is not None:
                    # verify greedy result on Node tree
                    if final_leaves(large_root, greedy_ds) != list(arr_tuple):
                        # inconsistent, discard greedy and fallback
                        greedy_ds = None

            if greedy_ds is not None:
                chosen_k = k
                chosen_array = arr_tuple
                chosen_del_small = set(next(iter(small_map[k][arr_tuple])))
                chosen_del_large = greedy_ds
                found = True
                reason = "greedy"
                break

            # Greedy failed (or disabled). Use cached backtracking result if available.
            cache_key = (tuple(large_arr), tuple(arr_tuple))
            if cache_key in backtrack_cache:
                back_ds = backtrack_cache[cache_key]
            else:
                # run backtracking with timeout
                back_ds = backtracking_with_timeout(large_arr, arr_tuple, timeout_sec=args.bt_timeout)
                backtrack_cache[cache_key] = back_ds

            if back_ds is not None:
                chosen_k = k
                chosen_array = arr_tuple
                chosen_del_small = set(next(iter(small_map[k][arr_tuple])))
                chosen_del_large = set(back_ds)
                found = True
                reason = "backtracking"
                break
            # else: try next candidate
        if found:
            break

    if not found:
        print("No common leaf-array found between the two trees for any k <= min(original leaf counts).")
        return

    # Map deletions back to A/B
    if small_name == "A":
        delA = chosen_del_small
        delB = chosen_del_large
    else:
        delA = chosen_del_large
        delB = chosen_del_small

    finalA = final_leaves(A, delA)
    finalB = final_leaves(B, delB)

    print(f"\nFound common leaf-array for k = {chosen_k} (matched via {reason} on larger tree):")
    print("Leaf-array:", list(chosen_array))
    print("Example deletions on Tree A:", sorted(delA) if delA else "(none)")
    print("Example deletions on Tree B:", sorted(delB) if delB else "(none)")
    print()

    # Print originals
    print("=== ORIGINAL Tree A ===")
    pretty_print_tree(A)
    print("Leaves:", leavesA)
    print()
    print("=== ORIGINAL Tree B ===")
    pretty_print_tree(B)
    print("Leaves:", leavesB)
    print()

    # Print finals
    print("=== FINAL Tree A (after deletions) ===")
    pretty_print_with_deletions(A, delA)
    print("Final leaves (A):", finalA)
    print()
    print("=== FINAL Tree B (after deletions) ===")
    pretty_print_with_deletions(B, delB)
    print("Final leaves (B):", finalB)
    print()

    # Visualization or save
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        draw_tree(A, deletions=set(), title="Original Tree A", save_path=os.path.join(save_dir, "orig_A.png"))
        draw_tree(B, deletions=set(), title="Original Tree B", save_path=os.path.join(save_dir, "orig_B.png"))
        draw_tree(A, deletions=delA, title=f"Final Tree A leaves={finalA}", save_path=os.path.join(save_dir, "final_A.png"))
        draw_tree(B, deletions=delB, title=f"Final Tree B leaves={finalB}", save_path=os.path.join(save_dir, "final_B.png"))
        print(f"Saved visualizations to {save_dir} (orig_A.png, orig_B.png, final_A.png, final_B.png)")
        return

    # Non-blocking displays (Option B)
    draw_tree(A, deletions=set(), title="Original Tree A")
    draw_tree(B, deletions=set(), title="Original Tree B")
    print("Opening original figures (you can move/resize them).")
    input("Press Enter to close originals and open final figures...")

    plt.close('all')
    draw_tree(A, deletions=delA, title=f"Final Tree A -> leaves={finalA}")
    draw_tree(B, deletions=delB, title=f"Final Tree B -> leaves={finalB}")
    print("Opening final figures (you can move/resize them).")
    input("Press Enter to close final figures and exit...")
    plt.close('all')

if __name__ == "__main__":
    # On Windows, multiprocessing requires the spawn-safe guard.
    main()
