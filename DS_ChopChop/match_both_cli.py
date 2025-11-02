#!/usr/bin/env python3
"""
match_both_cli.py

- Build two binary trees from level-order arrays (integers, None placeholders).
- Enumerate validated reachable leaf-arrays on the smaller tree (by original leaf count).
- Try to match candidate arrays on the other tree (backtracking).
- When a common array is found, print original/final trees and leaf lists, show plots
  (non-blocking: original pair shown together, then final pair shown together).
- CLI: --file arrays.txt (two non-empty lines), or --a and --b arrays directly.
- Optional: --save-dir to save PNGs instead of showing windows.
"""

from typing import Optional, List, Tuple, Set, Dict
from collections import defaultdict
from functools import lru_cache
import argparse
import ast
import os
import matplotlib.pyplot as plt

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
# Parsing arrays from CLI/file
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
# Basic printing and leaves
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
    # left
    if n.left is None:
        print(prefix + "  L-<none>")
    else:
        if f"{n.val}.L" in deletions:
            print(prefix + f"  L-<deleted edge to {n.left.val}>")
        else:
            pretty_print_with_deletions(n.left, deletions, prefix + "  L-")
    # right
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
# Validated enumeration: generate candidate (tuple, deletions) pairs, then validate by simulation
# --------------------
def gen_pairs(node: Optional[Node], memo=None):
    """
    Generate candidate (leaf_tuple, deletions_frozenset) pairs for subtree rooted at `node`.
    These pairs are candidates; later we validate them by applying deletions to the whole tree.
    """
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
    # leaf
    if node.left is None and node.right is None:
        results.add(((node.val,), frozenset()))
        memo[key] = results
        return results

    left_pairs = gen_pairs(node.left, memo) if node.left is not None else { ((), frozenset()) }
    right_pairs = gen_pairs(node.right, memo) if node.right is not None else { ((), frozenset()) }

    # keep both children -> concat
    for lt, ldel in left_pairs:
        for rt, rdel in right_pairs:
            tup = lt + rt
            deletions = set(ldel) | set(rdel)
            results.add((tup, frozenset(deletions)))

    # remove left child edge -> right supplies leaves
    if node.left is not None:
        for rt, rdel in right_pairs:
            tup = rt
            deletions = set(rdel) | {f"{node.val}.L"}
            results.add((tup, frozenset(deletions)))

    # remove right child edge -> left supplies leaves
    if node.right is not None:
        for lt, ldel in left_pairs:
            tup = lt
            deletions = set(ldel) | {f"{node.val}.R"}
            results.add((tup, frozenset(deletions)))

    # remove both -> node becomes leaf
    delset = set()
    if node.left is not None:
        delset.add(f"{node.val}.L")
    if node.right is not None:
        delset.add(f"{node.val}.R")
    results.add(((node.val,), frozenset(delset)))

    memo[key] = results
    return results

def build_validated_map(root: Optional[Node]):
    """
    Build map: k -> { leaf_tuple -> set(deletion_sets) }, keeping only pairs that actually produce
    the leaf_tuple when the deletions are applied to the whole tree.
    """
    raw_pairs = gen_pairs(root)
    mapk = defaultdict(lambda: defaultdict(set))
    for tup, ds in raw_pairs:
        # simulate
        simulated = tuple(final_leaves(root, set(ds)))
        if simulated == tup:
            mapk[len(tup)][tup].add(frozenset(ds))
    return mapk

# --------------------
# Match exact target on larger tree (backtracking)
# We return an example deletion-set; prefer one with smallest number of deleted edges.
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
        # returns list of (j, frozenset deletions)
        if node_id == 0:
            return [(i, frozenset())]
        node = nodes_by_id[node_id]
        results_local = []
        # if original leaf
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
                    # node becomes leaf
                    if i < n_target and target[i] == node.val:
                        ds = set()
                        if left_exists: ds.add(f"{node.val}.L")
                        if right_exists: ds.add(f"{node.val}.R")
                        results_local.append((i+1, frozenset(ds)))
                    continue
                # left matches
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
                    else:  # none
                        results_local.append((mid_idx, left_ds))
        return results_local

    matches = match(id(rootB), 0)
    # pick the match that consumes exactly n_target and has minimal deletion-set size
    valid_ds = [ds for (j, ds) in matches if j == n_target]
    if not valid_ds:
        return None
    # choose minimal deletions (tie-break lexicographically for determinism)
    best = min(valid_ds, key=lambda s: (len(s), sorted(s)))
    return set(best)

# --------------------
# Plot helpers (matplotlib)
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
    # draw edges
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
    # draw nodes
    for n, (x, y) in positions.items():
        if n in reachable:
            ax.scatter([x], [y], s=40)
            ax.text(x + 0.02, y, str(n.val), va='center', ha='left', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)  # non-blocking show for Option B

# --------------------
# Main CLI flow
# --------------------
def main():
    p = argparse.ArgumentParser(description="Match leaf-arrays between two trees by chopping edges.")
    p.add_argument('--file', '-f', help="Path to text file with two non-empty lines (arrays for A then B).")
    p.add_argument('--a', help="Array for tree A (python list or comma-separated), e.g. \"[1,2,3,None]\" or \"1,2,3,None\"")
    p.add_argument('--b', help="Array for tree B (python list or comma-separated).")
    p.add_argument('--save-dir', help="Optional directory to save visualization PNGs (will be created).")
    args = p.parse_args()

    if args.file:
        arrA, arrB = read_arrays_from_file(args.file)
    else:
        if not args.a or not args.b:
            print("Either --file or both --a and --b must be provided. See --help.")
            return
        arrA = parse_array_string(args.a)
        arrB = parse_array_string(args.b)

    A = build_tree_from_list(arrA)
    B = build_tree_from_list(arrB)

    leavesA = original_leaves(A)
    leavesB = original_leaves(B)
    print("Original Tree A leaves (L->R):", leavesA)
    print("Original Tree B leaves (L->R):", leavesB)
    print()

    # pick smaller by leaves to enumerate first
    if len(leavesA) <= len(leavesB):
        small_root, small_name, small_arr = A, "A", arrA
        large_root, large_name, large_arr = B, "B", arrB
    else:
        small_root, small_name, small_arr = B, "B", arrB
        large_root, large_name, large_arr = A, "A", arrA

    print(f"Enumerating validated reachable leaf-arrays on smaller tree (Tree {small_name})...")
    small_map = build_validated_map(small_root)

    # Search candidates descending by k, lexicographically over arrays for determinism
    start_k = max(small_map.keys()) if small_map else 0
    found = False
    chosen_k = None
    chosen_array = None
    chosen_del_small = None
    chosen_del_large = None

    for k in range(start_k, 0, -1):
        candidates = sorted(small_map.get(k, {}).keys())
        if not candidates:
            continue
        for arr_tuple in candidates:
            # try to match on larger tree (backtracking)
            del_large = find_deletions_to_match_exact(large_root, arr_tuple)
            if del_large is not None:
                # pick a preferred deletions set for the small tree (choose smallest # deletions)
                small_delsets = small_map[k][arr_tuple]
                chosen_small = min(small_delsets, key=lambda s: (len(s), sorted(s)))
                chosen_k = k
                chosen_array = arr_tuple
                chosen_del_small = set(chosen_small)
                chosen_del_large = set(del_large)
                found = True
                break
        if found:
            break

    if not found:
        print("No common leaf-array found between the two trees for any k <= min(original leaf counts).")
        return

    # Map deletions back to A/B depending on which was small
    if small_name == "A":
        delA = chosen_del_small
        delB = chosen_del_large
    else:
        delA = chosen_del_large
        delB = chosen_del_small

    finalA = final_leaves(A, delA)
    finalB = final_leaves(B, delB)

    # Report
    print(f"\nFound common leaf-array for k = {chosen_k}:")
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

    # Visualization: either save files or show non-blocking windows (Option B)
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        draw_tree(A, deletions=set(), title="Original Tree A", save_path=os.path.join(save_dir, "orig_A.png"))
        draw_tree(B, deletions=set(), title="Original Tree B", save_path=os.path.join(save_dir, "orig_B.png"))
        draw_tree(A, deletions=delA, title=f"Final Tree A leaves={finalA}", save_path=os.path.join(save_dir, "final_A.png"))
        draw_tree(B, deletions=delB, title=f"Final Tree B leaves={finalB}", save_path=os.path.join(save_dir, "final_B.png"))
        print(f"Saved visualizations to {save_dir} (orig_A.png, orig_B.png, final_A.png, final_B.png)")
        return

    # Show originals non-blocking, pause for user
    draw_tree(A, deletions=set(), title="Original Tree A")
    draw_tree(B, deletions=set(), title="Original Tree B")
    print("Opening original figures (you can move/resize them).")
    input("Press Enter to close originals and open final figures...")

    plt.close('all')
    # Show finals non-blocking, pause
    draw_tree(A, deletions=delA, title=f"Final Tree A -> leaves={finalA}")
    draw_tree(B, deletions=delB, title=f"Final Tree B -> leaves={finalB}")
    print("Opening final figures (you can move/resize them).")
    input("Press Enter to close final figures and exit...")
    plt.close('all')

if __name__ == "__main__":
    main()
