from collections import defaultdict, Counter
from datetime import datetime

def most_popular_transitions(lines):
    # Parse per-user events
    events = defaultdict(list)  # user -> [(ts, page), ...]
    for line in lines:
        ts_s, user, page = line.split()
        ts = datetime.fromisoformat(ts_s)
        events[user].append((ts, page))

    # Count transitions per user (consecutive visits)
    counts = Counter()
    for user, visits in events.items():
        visits.sort(key=lambda x: x[0])  # order by time per user
        for (t1, p1), (t2, p2) in zip(visits, visits[1:]):
            # If you want to ignore same-page repeats, uncomment next line:
            # if p1 == p2: continue
            counts[(p1, p2)] += 1

    if not counts:
        return [], 0

    max_count = max(counts.values())
    top = [(a, b) for (a, b), c in counts.items() if c == max_count]
    return top, max_count

# Example usage
lines = [
    "2020-01-10T12:00:00 user_1 page_a",
    "2020-01-10T12:01:00 user_1 page_b",
    "2020-01-10T12:05:00 user_2 page_c",
    "2020-01-10T12:07:00 user_1 page_c",
    "2020-01-10T12:09:00 user_2 page_b",
]

top, n = most_popular_transitions(lines)
print("Most popular transition count:", n)
for a, b in top:
    print(f"{a} -> {b}")
