"""Apriori mit Prefix-basiertem Self-Join. Skaliert auf viele Items.
Input: pro Event eine Menge von Anomalie-Keys. Output: Kombinationen die in
>= min_support_count Events gemeinsam auftreten."""

from collections import defaultdict


class ItemsetExplosion(Exception):
    """Kombinationsraum zu gross — Parameter (min_support_pct) zu niedrig."""


def apriori(transactions, min_support_count, max_size, max_candidates_per_level):
    """transactions: List[set]. Frequent-Items als kanonische Tuples (sortiert) verwaltet.
    Rueckgabe: dict[tuple, count] — alle Groessen 1..max_size.
    Wirft ItemsetExplosion wenn pro Level mehr als max_candidates_per_level Kandidaten entstehen."""
    if not transactions or min_support_count <= 0 or max_size < 1:
        return {}

    # L1: frequent 1-itemsets
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
    all_frequent = {}
    L1_items = []
    for item, cnt in item_counts.items():
        if cnt >= min_support_count:
            all_frequent[(item,)] = cnt
            L1_items.append(item)

    L_current = [(it,) for it in sorted(L1_items, key=lambda x: str(x))]
    k = 2
    while L_current and k <= max_size:
        # Gruppiere nach Prefix (ersten k-2 Elementen). Join nur innerhalb derselben Gruppe.
        groups = defaultdict(list)
        for t in L_current:
            groups[t[:-1]].append(t[-1])

        candidates = []
        for prefix, lasts in groups.items():
            lasts.sort(key=lambda x: str(x))
            for i in range(len(lasts)):
                for j in range(i + 1, len(lasts)):
                    cand = prefix + (lasts[i], lasts[j])
                    # Pruning: alle (k-1)-Subsets muessen frequent sein
                    valid = True
                    for skip_idx in range(k):
                        sub = cand[:skip_idx] + cand[skip_idx+1:]
                        if sub not in all_frequent:
                            valid = False
                            break
                    if valid:
                        candidates.append(cand)
                        if len(candidates) > max_candidates_per_level:
                            raise ItemsetExplosion(
                                f"Level {k}: >{max_candidates_per_level} Kandidaten — "
                                f"min_support_pct erhoehen oder max_set_size verringern")

        # Support zaehlen (als Frozenset fuer subset-Test)
        next_L = []
        for cand in candidates:
            cand_set = set(cand)
            cnt = 0
            for t in transactions:
                if cand_set.issubset(t):
                    cnt += 1
            if cnt >= min_support_count:
                all_frequent[cand] = cnt
                next_L.append(cand)

        L_current = next_L
        k += 1

    return all_frequent
