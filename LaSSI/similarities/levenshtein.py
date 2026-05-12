__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024,  Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__status__ = "Production"
from functools import lru_cache


def ld(s, t):
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if s[i - 1] == t[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
    return dp[n]


@lru_cache(maxsize=65536)
def lev(x, y):
    if len(x) == 0 and len(y) == 0:
        return 1.0
    L = max(len(x), len(y))
    return float(L - ld(x, y)) / float(L)


def MultiLevenshtein(x, y):
    L = y.split("\s+")
    overallDistance = 1.0
    for xX in x.split("\s+"):
        overallDistance *= lev(xX, min(L, key=lambda x: lev(x, xX)))
    return overallDistance
