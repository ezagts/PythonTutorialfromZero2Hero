# main.py
from ds_templates import (
    two_sum, length_of_longest_substring, first_ge,
    is_valid, next_greater, top_k,
    inorder, level_order, lca,
    bfs_shortest, length_of_LIS, coin_change, merge,
    uf_init, uf_find, uf_union, dijkstra
)


# Minimal TreeNode just for the tree demos
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def build_sample_tree():
    # LeetCode-style sample:
    #        3
    #      /   \
    #     5     1
    #    / \   / \
    #   6   2 0   8
    #      / \
    #     7   4
    n3 = TreeNode(3)
    n5 = TreeNode(5)
    n1 = TreeNode(1)
    n6 = TreeNode(6)
    n2 = TreeNode(2)
    n0 = TreeNode(0)
    n8 = TreeNode(8)
    n7 = TreeNode(7)
    n4 = TreeNode(4)
    n3.left, n3.right = n5, n1
    n5.left, n5.right = n6, n2
    n1.left, n1.right = n0, n8
    n2.left, n2.right = n7, n4
    return n3, n5, n1, n4  # root, and a couple of refs used by LCA


def main():
    print("=== Arrays / Hashing / Sliding Window / Binary Search ===")
    print("Two-Sum:", two_sum([2, 7, 11, 15], 9))  # -> [0,1]
    print("Longest substring w/o repeats:",
          length_of_longest_substring("pwwkew"))  # -> 3
    print("Binary search first >= x:",
          first_ge([1, 2, 4, 7], 5))  # -> 3

    print("\n=== Stack / Monotonic / Heap ===")
    print("Valid parentheses:", is_valid("({[]})"))  # -> True
    print("Valid parentheses:", is_valid("([)]"))  # -> False
    print("Next greater element:",
          next_greater([2, 1, 2, 4, 3]))  # -> [4,2,4,-1,-1]
    print("Top-K elements:", top_k([3, 1, 5, 12, 2, 11], 3))  # -> [12,11,5]

    print("\n=== Trees (DFS/BFS/LCA) ===")
    root, n5, n1, n4 = build_sample_tree()
    print("Inorder traversal:", inorder(root))  # -> [6,5,7,2,4,3,0,1,8]
    print("Level-order:", level_order(root))  # -> [[3],[5,1],[6,2,0,8],[7,4]]
    anc = lca(root, n5, n1)  # LCA(5,1) = 3
    print("LCA(5,1):", anc.val if anc else None)
    anc = lca(root, n5, n4)  # LCA(5,4) = 5
    print("LCA(5,4):", anc.val if anc else None)

    print("\n=== Graphs (BFS shortest paths / Dijkstra) ===")
    n = 4
    edges = [(0, 1), (0, 2), (1, 2), (1, 3)]  # undirected
    print("Unweighted shortest distances from 0:",
          bfs_shortest(n, edges, 0))  # -> [0,1,1,2]

    # Dijkstra adjacency list: list of lists of (a neighbor, weight)
    n2 = 5
    adj = [[] for _ in range(n2)]
    adj[0] = [(1, 2), (2, 5)]
    adj[1] = [(2, 1), (3, 2)]
    adj[2] = [(3, 3)]
    adj[3] = [(4, 1)]
    print("Dijkstra distances from 0:",
          dijkstra(n2, adj, 0))  # -> [0,2,3,4,5]

    print("\n=== Intervals / DP ===")
    print("Merge intervals:",
          merge([[1, 3], [2, 6], [8, 10], [15, 18]]))  # -> [[1,6],[8,10],[15,18]]
    print("Coin Change (min coins):",
          coin_change([1, 2, 5], 11))  # -> 3
    print("LIS length:",
          length_of_LIS([10, 9, 2, 5, 3, 7, 101, 18]))  # -> 4

    print("\n=== Union-Find (Disjoint Set) ===")
    p, sz = uf_init(5)  # 0..4
    uf_union(p, sz, 0, 1)
    uf_union(p, sz, 1, 2)
    uf_union(p, sz, 3, 4)
    # check connectivity (same root)
    same_0_2 = uf_find(p, 0) == uf_find(p, 2)
    same_0_3 = uf_find(p, 0) == uf_find(p, 3)
    print("Connected(0,2):", same_0_2)  # -> True
    print("Connected(0,3):", same_0_3)  # -> False


if __name__ == "__main__":
    main()
