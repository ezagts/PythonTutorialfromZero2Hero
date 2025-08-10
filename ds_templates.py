# def two_sum(nums, target):
#     seen = {}
#     for i, x in enumerate(nums):
#         if target - x in seen:
#             return [seen[target - x], i]
#         seen[x] = i
#     return []
#
#
# def length_of_longest_substring(s):
#     last, L, best = {}, 0, 0
#     for R, ch in enumerate(s):
#         if ch in last and last[ch] >= L:
#             L = last[ch] + 1
#         last[ch] = R
#         best = max(best, R - L + 1)
#     return best
#
#
# def first_ge(a, x):
#     L, R, ans = 0, len(a) - 1, len(a)
#     while L <= R:
#         m = (L + R) // 2
#         if a[m] >= x:
#             ans, R = m, m - 1
#         else:
#             L = m + 1
#     return ans
#
#
# def is_valid(s):
#     pairs = {')': '(', ']': '[', '}': '{'}
#     st = []
#     for ch in s:
#         if ch in '([{':
#             st.append(ch)
#         elif not st or st.pop() != pairs[ch]:
#             return False
#     return not st
#
#
# def next_greater(nums):
#     res = [-1] * len(nums)
#     st = []  # indices, values decreasing
#     for i, x in enumerate(nums):
#         while st and nums[st[-1]] < x:
#             res[st.pop()] = x
#         st.append(i)
#     return res
#
#
# import heapq
#
#
# def top_k(nums, k):
#     return heapq.nlargest(k, nums)
#
#
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val, self.next = val, next
#
#
# def reverse_list(head):
#     prev, cur = None, head
#     while cur:
#         cur.next, prev, cur = prev, cur, cur.next
#     return prev
#
#
# def has_cycle(head):
#     slow = fast = head
#     while fast and fast.next:
#         slow, fast = slow.next, fast.next.next
#         if slow is fast: return True
#     return False
#
#
# def inorder(root):
#     out = []
#
#     def dfs(node):
#         if not node: return
#         dfs(node.left);
#         out.append(node.val);
#         dfs(node.right)
#
#     dfs(root);
#     return out
#
#
# from collections import deque
#
#
# def level_order(root):
#     if not root: return []
#     q, levels = deque([root]), []
#     while q:
#         size, cur = len(q), []
#         for _ in range(size):
#             node = q.popleft();
#             cur.append(node.val)
#             if node.left: q.append(node.left)
#             if node.right: q.append(node.right)
#         levels.append(cur)
#     return levels
#
#
# def lca(root, p, q):
#     if not root or root is p or root is q: return root
#     L, R = lca(root.left, p, q), lca(root.right, p, q)
#     return root if L and R else L or R
#
#
# from collections import defaultdict, deque
#
#
# def bfs_shortest(n, edges, src):
#     g = defaultdict(list)
#     for u, v in edges:
#         g[u].append(v);
#         g[v].append(u)
#     dist = [float('inf')] * n
#     dist[src] = 0
#     q = deque([src])
#     while q:
#         u = q.popleft()
#         for v in g[u]:
#             if dist[v] == float('inf'):
#                 dist[v] = dist[u] + 1
#                 q.append(v)
#     return dist
#
#
# import heapq
#
#
# def dijkstra(n, adj, src):
#     dist = [float('inf')] * n
#     dist[src] = 0
#     pq = [(0, src)]
#     while pq:
#         d, u = heapq.heappop(pq)
#         if d != dist[u]: continue
#         for v, w in adj[u]:
#             nd = d + w
#             if nd < dist[v]:
#                 dist[v] = nd
#                 heapq.heappush(pq, (nd, v))
#     return dist
#
#
# def uf_init(n): return list(range(n)), [1] * n
#
#
# def uf_find(p, x):
#     if p[x] != x: p[x] = uf_find(p, p[x])
#     return p[x]
#
#
# def uf_union(p, sz, a, b):
#     ra, rb = uf_find(p, a), uf_find(p, b)
#     if ra == rb: return False
#     if sz[ra] < sz[rb]: ra, rb = rb, ra
#     p[rb] = ra;
#     sz[ra] += sz[rb];
#     return True
#
#
# def merge(intervals):
#     intervals.sort()
#     out = []
#     for s, e in intervals:
#         if not out or s > out[-1][1]:
#             out.append([s, e])
#         else:
#             out[-1][1] = max(out[-1][1], e)
#     return out
#
#
# def coin_change(coins, amount):
#     INF = 10 ** 9
#     dp = [0] + [INF] * amount
#     for a in range(1, amount + 1):
#         for c in coins:
#             if c <= a and dp[a - c] + 1 < dp[a]:
#                 dp[a] = dp[a - c] + 1
#     return -1 if dp[amount] >= INF else dp[amount]
#
#
# import bisect
#
#
# def length_of_LIS(nums):
#     tails = []
#     for x in nums:
#         i = bisect.bisect_left(tails, x)
#         if i == len(tails):
#             tails.append(x)
#         else:
#             tails[i] = x
#     return len(tails)

# ds_templates.py
from typing import List, Tuple, Dict, Optional
import heapq
import bisect
from collections import defaultdict, deque


# ---------- Arrays / Hashing ----------

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Return indices [i, j] such that nums[i] + nums[j] == target (the first pair found).
    Uses a hash map of value->index. O(n) time, O(n) space.

    """
    seen: Dict[int, int] = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []


def length_of_longest_substring(s: str) -> int:
    """

    Sliding window: length of longest substring without repeating characters.
    O(n) time, O(min(n, alphabet)) space.

    """
    last: Dict[str, int] = {}
    L = best = 0
    for R, ch in enumerate(s):
        if ch in last and last[ch] >= L:
            L = last[ch] + 1
        last[ch] = R
        best = max(best, R - L + 1)
    return best


def first_ge(a: List[int], x: int) -> int:
    """

    Binary search: index of the first element >= x in a sorted list `a`.
    Returns len(a) if none. O(log n).

    """
    L, R, ans = 0, len(a) - 1, len(a)
    while L <= R:
        m = (L + R) // 2
        if a[m] >= x:
            ans = m
            R = m - 1
        else:
            L = m + 1
    return ans


# ---------- Stack / Monotonic / Heap ----------

def is_valid(s: str) -> bool:
    """

    Valid parentheses: (), [], {} must be properly nested/closed. O(n).

    """
    pairs = {')': '(', ']': '[', '}': '{'}
    st: List[str] = []
    for ch in s:
        if ch in '([{':
            st.append(ch)
        else:
            if not st or st.pop() != pairs.get(ch, '#'):
                return False
    return not st


def next_greater(nums: List[int]) -> List[int]:
    """

    For each element, returns the next greater element to its right or -1 if none.
    Monotonic decreasing stack of indices. O(n).

    """
    res = [-1] * len(nums)
    st: List[int] = []
    for i, x in enumerate(nums):
        while st and nums[st[-1]] < x:
            res[st.pop()] = x
        st.append(i)
    return res


def top_k(nums: List[int], k: int) -> List[int]:
    """

    Returns the k largest elements (descending) using heapq.nlargest.
    O(n log k) average.

    """
    return heapq.nlargest(k, nums)


# ---------- Linked List helpers (optional for your set) ----------

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val, self.next = val, next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative list reversal. O(n) time, O(1) space.
    """
    prev, cur = None, head
    while cur:
        cur.next, prev, cur = prev, cur, cur.next
    return prev


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Floyd's cycle detection (tortoise/hare). O(n) time, O(1) space.
    """
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False


# ---------- Trees ----------

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val, self.left, self.right = val, left, right


def inorder(root: Optional[TreeNode]) -> List[int]:
    """
    Recursive inorder traversal (Left, Node, Right). Returns list of values. O(n).
    """
    out: List[int] = []

    def dfs(node: Optional[TreeNode]) -> None:
        if not node: return
        dfs(node.left)
        out.append(node.val)
        dfs(node.right)

    dfs(root)
    return out


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS level-order traversal. Returns list of levels. O(n).
    """
    if not root: return []
    q: deque[TreeNode] = deque([root])
    levels: List[List[int]] = []
    while q:
        size = len(q)
        cur: List[int] = []
        for _ in range(size):
            node = q.popleft()
            cur.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        levels.append(cur)
    return levels


def lca(root: Optional[TreeNode],
        p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
    """
    Lowest Common Ancestor in a general binary tree.
    Assumes p and q are nodes from the same tree. O(n).
    """
    if not root or root is p or root is q:
        return root
    L = lca(root.left, p, q)
    R = lca(root.right, p, q)
    return root if L and R else (L or R)


# ---------- Graphs ----------

def bfs_shortest(n: int, edges: List[Tuple[int, int]], src: int) -> List[float]:
    """
    Unweighted shortest-path distances from src in an undirected graph with n nodes (0..n-1).
    Returns a list of distances (float('inf') if unreachable). O(n + m).
    """
    g: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)

    dist = [float('inf')] * n
    dist[src] = 0
    q: deque[int] = deque([src])

    while q:
        u = q.popleft()
        for v in g[u]:
            if dist[v] == float('inf'):
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def dijkstra(n: int, adj: List[List[Tuple[int, int]]], src: int) -> List[float]:
    """
    Dijkstra for non-negative edge weights.
    adj[u] = list of (v, w). Returns distances from src. O(m log n).
    """
    dist = [float('inf')] * n
    dist[src] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:  # stale
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ---------- Intervals ----------

def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge overlapping [start, end] intervals. O(n log n) for sort, O(1) extra.
    """
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0][:]]
    for s, e in intervals[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out


# ---------- Dynamic Programming ----------

def coin_change(coins: List[int], amount: int) -> int:
    """
    Fewest coins to make `amount`; returns -1 if impossible.
    Bottom-up DP. O(amount * len(coins)).
    """
    INF = 10 ** 9
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        best = INF
        for c in coins:
            if c <= a and dp[a - c] + 1 < best:
                best = dp[a - c] + 1
        dp[a] = best
    return -1 if dp[amount] >= INF else dp[amount]


def length_of_LIS(nums: List[int]) -> int:
    """
    Length of the Longest Increasing Subsequence.
    Patience sorting w/ binary search. O(n log n).
    """
    tails: List[int] = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


# ---------- Union-Find (Disjoint Set) ----------

def uf_init(n: int) -> Tuple[List[int], List[int]]:
    """
    Initialize parent and size arrays for n elements (0..n-1).
    """
    return list(range(n)), [1] * n


def uf_find(p: List[int], x: int) -> int:
    """
    Find with path compression. Amortized almost O(1).
    """
    if p[x] != x:
        p[x] = uf_find(p, p[x])
    return p[x]


def uf_union(p: List[int], sz: List[int], a: int, b: int) -> bool:
    """
    Union by size; returns True if merged, False if already in same set.
    """
    ra, rb = uf_find(p, a), uf_find(p, b)
    if ra == rb:
        return False
    if sz[ra] < sz[rb]:
        ra, rb = rb, ra
    p[rb] = ra
    sz[ra] += sz[rb]
    return True
