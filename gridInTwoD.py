"""
You are given a 2D grid representation of the ocean.
There are a numerous land and ocean cells as part of this grid.

Q1: Given this representation, return the number of land cells.
Q2: Given this representation, return the number of islands in the grid.

"""

from collections import deque


def count_land_cells(grid, land='1'):
    """Q1: Count land cells."""
    if not grid: return 0
    return sum(1 for r in grid for c in r if c == land)


def num_islands(grid, land='1'):
    """Q2: Count islands (4-directional)."""
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    visited = [[False] * C for _ in range(R)]

    def bfs(sr, sc):
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == land:
                    visited[nr][nc] = True
                    q.append((nr, nc))

    islands = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == land and not visited[r][c]:
                islands += 1
                bfs(r, c)
    return islands


# Example
grid = [
    "1100",
    "1101",
    "0010",
    "0011",
]
print("Land cells:", count_land_cells(grid))  # -> 7
print("Islands:", num_islands(grid))  # -> 3
