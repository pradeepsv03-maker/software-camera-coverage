"""
Software camera coverage check  (Python 3.10+, standard library only).

PROBLEM
    A "software camera" is built from many fixed hardware cameras. Each hardware
    camera works for a range of subject distances AND a range of light levels.
    Given the ranges the software camera must support and a list of hardware
    cameras, decide whether the cameras together support every
    (distance, light) combination.

MODEL
    Every camera is a closed rectangle in the (distance, light) plane:
        [dist_min, dist_max] x [light_min, light_max]
    The requirement is such a rectangle too. The answer is True exactly when the
    UNION of the camera rectangles contains the WHOLE required rectangle.

    Decisions (the statement leaves them open, so they are explicit):
      * Ranges are CLOSED and CONTINUOUS. Distance 5 belongs to both [0, 5] and
        [5, 10], so those two cameras together cover [0, 10]. But [0, 5] and
        [5.0001, 10] leave a gap, which is a failure.
      * Two cameras that only touch at a corner do not cover the areas around
        that corner (a diagonal touch is not a cover).
      * A range with min == max is legal (a single value).
      * min > max, NaN, non-numbers or malformed cameras raise an error rather
        than silently returning a wrong answer. +/- infinity is allowed
        (e.g. "works for any light level above 100").
      * Comparisons are exact. Use ints or fractions.Fraction if you need exact
        decimal boundaries (floats such as 0.1 + 0.2 != 0.3 behave as floats do).

ALGORITHM  (rectangle-union coverage: coordinate compression + sweep line +
            segment tree)   Time O(n log n), space O(n)
    1. Validate input; discard cameras that do not touch the required rectangle.
    2. Coordinate compression with "doubling". On each axis take the distinct
       camera boundaries inside the required range (plus the range ends) and
       give every boundary an EVEN index 2k. The open gap between boundary k
       and k+1 gets the ODD index 2k+1. Every elementary piece of the axis (a
       single point or an open gap) is now one integer cell, and a closed range
       [a, b] is exactly the cell range 2*idx(a) .. 2*idx(b). Inside one cell
       the set of covering cameras never changes, so checking cells is exact
       (no sampling, no epsilon, no special cases for touching edges).
    3. Sweep over the distance cells. A camera adds +1 on its light-cell range
       when the sweep reaches its first distance cell and -1 just after its
       last one. A segment tree (range add, global minimum, lazy-free) keeps the
       minimum coverage over all light cells.
    4. After each batch of events the coverage is constant until the next batch,
       so it is enough to check that the minimum is > 0 there. The first batch
       must start at the first distance cell. Any check that fails means an
       uncovered cell exists -> False. If all pass -> True.

    Sorting/compression O(n log n); 2n range updates at O(log n) each.

USAGE
    from software_camera import Camera, can_build_software_camera
    ok = can_build_software_camera(
        required_distance=(1, 10),                 # (min, max)
        required_light=(0, 100),                   # (min, max)
        cameras=[Camera(1, 5, 0, 100), Camera(5, 10, 0, 100)],
    )                                              # -> True
    Plain 4-tuples (dist_min, dist_max, light_min, light_max) also work.

    Run the tests:   python software_camera.py
"""

from __future__ import annotations

import math
import numbers
from bisect import bisect_left
from typing import Iterable, List, NamedTuple, Sequence, Tuple


class Camera(NamedTuple):
    """A hardware camera: usable distance range and usable light-level range."""

    dist_min: float
    dist_max: float
    light_min: float
    light_max: float


def _check_range(name: str, lo: float, hi: float) -> None:
    for value in (lo, hi):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name}: bounds must be real numbers, got {value!r}")
        if value != value:  # NaN
            raise ValueError(f"{name}: bounds must not be NaN")
    if lo > hi:
        raise ValueError(f"{name}: minimum {lo!r} is greater than maximum {hi!r}")


def _unpack_range(name: str, rng: Sequence[float]) -> Tuple[float, float]:
    try:
        lo, hi = rng
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a (min, max) pair, got {rng!r}") from None
    _check_range(name, lo, hi)
    return lo, hi


def _compress(lo: float, hi: float, values: Iterable[float]) -> List[float]:
    """Sorted distinct coordinates: lo, hi and every value strictly inside."""
    coords = {lo, hi}
    coords.update(v for v in values if lo < v < hi)
    return sorted(coords)


class _MinCoverTree:
    """
    Range-add / global-minimum segment tree over `size` cells (bottom-up, no
    lazy propagation). tree[p] = min over its children + pending[p], so the
    global minimum is simply tree[1].
    Cells beyond `size` (power-of-two padding) hold +inf and are never touched.
    """

    def __init__(self, size: int) -> None:
        n = 1
        while n < size:
            n *= 2
        self._n = n
        self._tree = [0] * n + [0] * size + [math.inf] * (n - size)
        # Internal nodes must start as the min of their children.
        for p in range(n - 1, 0, -1):
            self._tree[p] = min(self._tree[2 * p], self._tree[2 * p + 1])
        self._pending = [0] * n

    def add(self, left: int, right: int, delta: int) -> None:
        """Add `delta` to every cell in the inclusive range [left, right]."""
        n, tree, pending = self._n, self._tree, self._pending
        l, r = left + n, right + n + 1
        l0, r0 = l, r
        while l < r:
            if l & 1:
                tree[l] += delta
                if l < n:
                    pending[l] += delta
                l += 1
            if r & 1:
                r -= 1
                tree[r] += delta
                if r < n:
                    pending[r] += delta
            l >>= 1
            r >>= 1
        for p in (l0, r0 - 1):
            p >>= 1
            while p:
                a, b = tree[2 * p], tree[2 * p + 1]
                tree[p] = (a if a < b else b) + pending[p]
                p >>= 1

    def minimum(self) -> float:
        return self._tree[1]


def can_build_software_camera(
    required_distance: Sequence[float],
    required_light: Sequence[float],
    cameras: Iterable[Sequence[float]],
) -> bool:
    """
    Return True if the hardware `cameras` together support every subject
    distance in `required_distance` combined with every light level in
    `required_light` (both are closed (min, max) ranges).

    Each camera is a Camera or a (dist_min, dist_max, light_min, light_max)
    sequence. Raises TypeError/ValueError on malformed input.
    Runs in O(n log n) time and O(n) space for n cameras.
    """
    dist_lo, dist_hi = _unpack_range("required_distance", required_distance)
    light_lo, light_hi = _unpack_range("required_light", required_light)

    # Validate every camera (even irrelevant ones, so bad data never hides),
    # and keep only those that touch the required rectangle.
    relevant: List[Camera] = []
    for index, raw in enumerate(cameras):
        try:
            cam = Camera(*raw)
        except TypeError:
            raise TypeError(
                f"camera #{index} must be (dist_min, dist_max, light_min, light_max), "
                f"got {raw!r}"
            ) from None
        _check_range(f"camera #{index} distance", cam.dist_min, cam.dist_max)
        _check_range(f"camera #{index} light", cam.light_min, cam.light_max)
        if (
            cam.dist_max >= dist_lo
            and cam.dist_min <= dist_hi
            and cam.light_max >= light_lo
            and cam.light_min <= light_hi
        ):
            relevant.append(cam)

    if not relevant:
        return False  # the required area is never empty (min <= max)

    # Coordinate compression with doubling: boundary k -> cell 2k, gap -> 2k+1.
    d_coords = _compress(
        dist_lo, dist_hi, (v for c in relevant for v in (c.dist_min, c.dist_max))
    )
    l_coords = _compress(
        light_lo, light_hi, (v for c in relevant for v in (c.light_min, c.light_max))
    )
    last_d_cell = 2 * (len(d_coords) - 1)
    light_cells = 2 * len(l_coords) - 1

    def cell_range(lo: float, hi: float, coords: List[float]) -> Tuple[int, int]:
        # Clamp to the required range; an outside bound covers from/to the end.
        first = 0 if lo <= coords[0] else 2 * bisect_left(coords, lo)
        last = 2 * (len(coords) - 1) if hi >= coords[-1] else 2 * bisect_left(coords, hi)
        return first, last

    # Events: (distance cell, light first, light last, +1 / -1).
    events: List[Tuple[int, int, int, int]] = []
    for cam in relevant:
        d_first, d_last = cell_range(cam.dist_min, cam.dist_max, d_coords)
        l_first, l_last = cell_range(cam.light_min, cam.light_max, l_coords)
        events.append((d_first, l_first, l_last, 1))
        if d_last < last_d_cell:  # removal beyond the last cell is irrelevant
            events.append((d_last + 1, l_first, l_last, -1))
    events.sort()

    # The very first distance cell must be reached by some camera.
    if events[0][0] != 0:
        return False

    tree = _MinCoverTree(light_cells)
    i, total = 0, len(events)
    while i < total:
        cell = events[i][0]
        while i < total and events[i][0] == cell:
            _, l_first, l_last, delta = events[i]
            tree.add(l_first, l_last, delta)
            i += 1
        # Coverage is constant from `cell` until the next batch (or the end).
        if tree.minimum() <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Tests (standard library `unittest`; run with:  python software_camera.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import ast
    import inspect
    import itertools
    import random
    import sys
    import time
    import unittest
    from fractions import Fraction

    INF = math.inf

    def ok(req_d, req_l, cams):
        return can_build_software_camera(req_d, req_l, cams)

    class BasicCoverage(unittest.TestCase):
        def test_single_camera_exact_match(self):
            self.assertTrue(ok((0, 10), (0, 10), [(0, 10, 0, 10)]))

        def test_single_camera_larger_than_needed(self):
            self.assertTrue(ok((2, 3), (2, 3), [(0, 10, 0, 10)]))

        def test_single_camera_too_small_each_side(self):
            for cam in [(1, 10, 0, 10), (0, 9, 0, 10), (0, 10, 1, 10), (0, 10, 0, 9)]:
                self.assertFalse(ok((0, 10), (0, 10), [cam]), cam)

        def test_empty_camera_list(self):
            self.assertFalse(ok((0, 10), (0, 10), []))

        def test_only_irrelevant_cameras(self):
            self.assertFalse(ok((0, 10), (0, 10), [(20, 30, 0, 10), (0, 10, 50, 60)]))

        def test_two_cameras_split_by_distance_touching(self):
            self.assertTrue(ok((0, 10), (0, 10), [(0, 5, 0, 10), (5, 10, 0, 10)]))

        def test_two_cameras_split_by_light_touching(self):
            self.assertTrue(ok((0, 10), (0, 10), [(0, 10, 0, 5), (0, 10, 5, 10)]))

        def test_two_cameras_overlapping(self):
            self.assertTrue(ok((0, 10), (0, 10), [(0, 6, 0, 10), (4, 10, 0, 10)]))

        def test_gap_between_cameras_distance(self):
            self.assertFalse(ok((0, 10), (0, 10), [(0, 4, 0, 10), (5, 10, 0, 10)]))

        def test_gap_between_cameras_light(self):
            self.assertFalse(ok((0, 10), (0, 10), [(0, 10, 0, 4), (0, 10, 5, 10)]))

        def test_tiny_gap_is_a_gap(self):
            eps = Fraction(1, 10**9)
            self.assertFalse(
                ok((0, 10), (0, 10), [(0, 5, 0, 10), (5 + eps, 10, 0, 10)])
            )

        def test_four_tiles_cover(self):
            tiles = [(0, 5, 0, 5), (5, 10, 0, 5), (0, 5, 5, 10), (5, 10, 5, 10)]
            self.assertTrue(ok((0, 10), (0, 10), tiles))

        def test_missing_one_tile_fails(self):
            tiles = [(0, 5, 0, 5), (5, 10, 0, 5), (0, 5, 5, 10)]
            self.assertFalse(ok((0, 10), (0, 10), tiles))

        def test_diagonal_corner_touch_is_not_a_cover(self):
            self.assertFalse(ok((0, 10), (0, 10), [(0, 5, 0, 5), (5, 10, 5, 10)]))

        def test_donut_hole_in_the_middle(self):
            ring = [
                (0, 10, 0, 3), (0, 10, 7, 10),  # bottom / top bars
                (0, 3, 3, 7), (7, 10, 3, 7),  # left / right bars
            ]
            self.assertFalse(ok((0, 10), (0, 10), ring))
            self.assertTrue(ok((0, 10), (0, 10), ring + [(3, 7, 3, 7)]))

        def test_hole_that_only_touches_edges_of_pieces(self):
            # Uncovered single point (5, 5): the four quadrants leave it out.
            quads = [(0, 5, 0, 5), (5, 10, 0, 5), (0, 5, 5, 10), (5, 10, 5, 10)]
            self.assertTrue(ok((0, 10), (0, 10), quads))
            # Shrink by removing shared boundary coverage -> real gap cross.
            self.assertFalse(
                ok((0, 10), (0, 10), [(0, 4, 0, 4), (6, 10, 0, 4), (0, 4, 6, 10), (6, 10, 6, 10)])
            )

        def test_staircase_of_overlapping_cameras(self):
            cams = [(i, i + 2, i, i + 2) for i in range(0, 9)]
            self.assertFalse(ok((0, 10), (0, 10), cams))  # diagonal band only
            band_plus = cams + [(0, 10, 0, 1)]
            self.assertFalse(ok((0, 10), (0, 10), band_plus))

        def test_nested_and_duplicate_cameras(self):
            cams = [(0, 10, 0, 10)] * 3 + [(2, 3, 2, 3)]
            self.assertTrue(ok((0, 10), (0, 10), cams))

        def test_order_of_cameras_does_not_matter(self):
            cams = [(0, 4, 0, 10), (4, 8, 0, 10), (8, 12, 0, 10)]
            for perm in itertools.permutations(cams):
                self.assertTrue(ok((0, 12), (0, 10), perm))

        def test_camera_with_extra_range_outside_requirement(self):
            self.assertTrue(ok((5, 6), (5, 6), [(0, 5.5, 0, 100), (5.5, 100, 0, 100)]))

        def test_realistic_example(self):
            # distance in meters, light in lux
            cams = [
                Camera(0.1, 2, 1, 1000),        # macro, low..mid light
                Camera(2, 50, 1, 200),          # normal, low..mid light
                Camera(2, 50, 200, 100000),     # normal, bright light
                Camera(0.1, 2, 1000, 100000),   # macro, bright light
                Camera(50, INF, 1, 100000),     # far
            ]
            self.assertTrue(ok((0.1, 1000), (1, 100000), cams))
            self.assertFalse(ok((0.1, 1000), (1, 100000), cams[:-1]))
            self.assertFalse(ok((0.1, 1000), (0.5, 100000), cams))  # too dark

    class DegenerateAndSpecialValues(unittest.TestCase):
        def test_point_requirement(self):
            self.assertTrue(ok((5, 5), (5, 5), [(0, 10, 0, 10)]))
            self.assertTrue(ok((5, 5), (5, 5), [(5, 5, 5, 5)]))
            self.assertFalse(ok((5, 5), (5, 5), [(6, 7, 5, 5)]))

        def test_line_requirement(self):
            self.assertTrue(ok((5, 5), (0, 10), [(5, 5, 0, 4), (0, 9, 4, 10)]))
            self.assertFalse(ok((5, 5), (0, 10), [(5, 5, 0, 4), (0, 9, 5, 10)]))

        def test_point_camera_cannot_cover_area(self):
            self.assertFalse(ok((0, 1), (0, 1), [(0, 0, 0, 0)]))

        def test_zero_width_camera_bridges_touching_pieces(self):
            self.assertTrue(
                ok((0, 10), (0, 10), [(0, 5, 0, 10), (5, 5, 0, 10), (5, 10, 0, 10)])
            )

        def test_negative_and_float_values(self):
            self.assertTrue(ok((-5.5, 5.5), (-1.0, 1.0), [(-6, 0, -2, 2), (0, 6, -2, 2)]))

        def test_infinite_bounds(self):
            self.assertTrue(ok((0, INF), (0, INF), [(0, INF, 0, INF)]))
            self.assertTrue(ok((0, INF), (0, 10), [(0, 5, 0, 10), (5, INF, 0, 10)]))
            self.assertFalse(ok((0, INF), (0, 10), [(0, 5, 0, 10), (5, 1e300, 0, 10)]))
            self.assertTrue(ok((-INF, INF), (-INF, INF), [(-INF, 0, -INF, INF), (0, INF, -INF, INF)]))

        def test_fractions_are_exact(self):
            third = Fraction(1, 3)
            self.assertTrue(ok((0, 1), (0, 1), [(0, third, 0, 1), (third, 1, 0, 1)]))

        def test_float_boundaries_are_compared_exactly(self):
            # documented behaviour: 0.1 + 0.2 is 0.30000000000000004, not 0.3
            self.assertFalse(ok((0, 1), (0, 1), [(0, 0.3, 0, 1), (0.1 + 0.2, 1, 0, 1)]))
            self.assertTrue(ok((0, 1), (0, 1), [(0, 0.3, 0, 1), (0.3, 1, 0, 1)]))

        def test_accepts_generator_and_does_not_mutate_input(self):
            cams = [(5, 10, 0, 10), (0, 5, 0, 10)]
            snapshot = list(cams)
            self.assertTrue(ok((0, 10), (0, 10), (c for c in cams)))
            self.assertEqual(cams, snapshot)

        def test_accepts_camera_namedtuples_and_lists(self):
            self.assertTrue(ok([0, 10], [0, 10], [Camera(0, 10, 0, 10)]))
            self.assertTrue(ok((0, 10), (0, 10), [[0, 10, 0, 10]]))

    class InvalidInput(unittest.TestCase):
        def test_min_greater_than_max(self):
            with self.assertRaises(ValueError):
                ok((10, 0), (0, 10), [(0, 10, 0, 10)])
            with self.assertRaises(ValueError):
                ok((0, 10), (10, 0), [(0, 10, 0, 10)])
            with self.assertRaises(ValueError):
                ok((0, 10), (0, 10), [(10, 0, 0, 10)])
            with self.assertRaises(ValueError):
                ok((0, 10), (0, 10), [(0, 10, 10, 0)])

        def test_invalid_camera_is_reported_even_if_irrelevant(self):
            with self.assertRaises(ValueError):
                ok((0, 10), (0, 10), [(0, 10, 0, 10), (50, 40, 0, 1)])

        def test_slightly_inverted_ranges_are_rejected(self):
            for args in [
                ((10, 9), (0, 10), [(0, 10, 0, 10)]),
                ((0, 10), (10, 9.999), [(0, 10, 0, 10)]),
                ((0, 10), (0, 10), [(10, 9.5, 0, 10)]),
                ((0, 10), (0, 10), [(0, 10, 3, 2.5)]),
                ((Fraction(1, 3), Fraction(1, 4)), (0, 1), []),
            ]:
                with self.assertRaises(ValueError):
                    ok(*args)

        def test_nan(self):
            nan = float("nan")
            with self.assertRaises(ValueError):
                ok((nan, 10), (0, 10), [])
            with self.assertRaises(ValueError):
                ok((0, 10), (0, 10), [(0, nan, 0, 10)])

        def test_wrong_types(self):
            for bad in ["a", None, True]:
                with self.assertRaises(TypeError):
                    ok((bad, 10), (0, 10), [])
                with self.assertRaises(TypeError):
                    ok((0, 10), (0, 10), [(0, bad, 0, 10)])

        def test_malformed_shapes(self):
            with self.assertRaises(TypeError):
                ok(5, (0, 10), [])
            with self.assertRaises(TypeError):
                ok((0, 10, 3), (0, 10), [])
            with self.assertRaises(TypeError):
                ok((0, 10), (0, 10), [(0, 10, 0)])
            with self.assertRaises(TypeError):
                ok((0, 10), (0, 10), [(0, 10, 0, 10, 99)])
            with self.assertRaises(TypeError):
                ok((0, 10), (0, 10), [42])

    class BruteForceCrossCheck(unittest.TestCase):
        """
        Independent oracle: on an integer grid, double every coordinate so that
        even indices are points and odd indices are the open gaps between them.
        A closed camera [lo, hi] covers doubled indices 2*lo .. 2*hi inclusive.
        The requirement is met iff every doubled cell is covered. This shares no
        code with the real implementation.
        """

        @staticmethod
        def oracle(req_d, req_l, cams):
            covered = set()
            for d0, d1, l0, l1 in cams:
                for i in range(2 * d0, 2 * d1 + 1):
                    for j in range(2 * l0, 2 * l1 + 1):
                        covered.add((i, j))
            return all(
                (i, j) in covered
                for i in range(2 * req_d[0], 2 * req_d[1] + 1)
                for j in range(2 * req_l[0], 2 * req_l[1] + 1)
            )

        @staticmethod
        def random_case(rng, size):
            def rng_pair():
                a, b = rng.randint(0, size), rng.randint(0, size)
                return min(a, b), max(a, b)

            req_d, req_l = rng_pair(), rng_pair()
            cams = [rng_pair() + rng_pair() for _ in range(rng.randint(0, 7))]
            return req_d, req_l, cams

        def test_matches_oracle_on_random_cases(self):
            rng = random.Random(20240607)
            trues = falses = 0
            for _ in range(6000):
                req_d, req_l, cams = self.random_case(rng, 6)
                expected = self.oracle(req_d, req_l, cams)
                self.assertEqual(ok(req_d, req_l, cams), expected, (req_d, req_l, cams))
                trues += expected
                falses += not expected
            # make sure the random test really exercises both outcomes
            self.assertGreater(trues, 300)
            self.assertGreater(falses, 300)

        def test_result_invariant_under_scaling_and_shifting(self):
            rng = random.Random(7)
            for _ in range(1500):
                req_d, req_l, cams = self.random_case(rng, 6)
                base = ok(req_d, req_l, cams)
                fd = lambda v: Fraction(v) * Fraction(7, 3) - 11
                fl = lambda v: v * 0.5 + 100  # exactly representable in binary
                moved = ok(
                    tuple(map(fd, req_d)),
                    tuple(map(fl, req_l)),
                    [(fd(a), fd(b), fl(c), fl(d)) for a, b, c, d in cams],
                )
                self.assertEqual(base, moved)

        def test_adding_cameras_never_breaks_coverage(self):
            rng = random.Random(99)
            for _ in range(1500):
                req_d, req_l, cams = self.random_case(rng, 6)
                extra = self.random_case(rng, 6)[2]
                if ok(req_d, req_l, cams):
                    self.assertTrue(ok(req_d, req_l, cams + extra))

    class Performance(unittest.TestCase):
        def test_2000_cameras_finish_quickly(self):
            n = 2000
            # overlapping chain that covers [0, n] fully
            cams = [(i, i + 2, 0, 10) for i in range(n)]
            start = time.perf_counter()
            self.assertTrue(ok((0, n + 1), (0, 10), cams))
            cams.pop(n // 2)  # remove one link -> still covered by overlap
            self.assertTrue(ok((0, n + 1), (0, 10), cams))
            cams = [c for c in cams if not (n // 2 - 1 <= c[0] <= n // 2 + 1)]  # real gap
            self.assertFalse(ok((0, n + 1), (0, 10), cams))
            self.assertLess(time.perf_counter() - start, 15)

        def test_grid_of_tiles(self):
            k = 40
            tiles = [(i, i + 1, j, j + 1) for i in range(k) for j in range(k)]
            self.assertTrue(ok((0, k), (0, k), tiles))
            tiles.remove((17, 18, 23, 24))
            self.assertFalse(ok((0, k), (0, k), tiles))

    class TaskRequirements(unittest.TestCase):
        """The constraints from the assignment text, checked mechanically."""

        def test_standard_library_only(self):
            tree = ast.parse(inspect.getsource(sys.modules["__main__"]))
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names.add(node.module.split(".")[0])
            third_party = {n for n in names if n not in sys.stdlib_module_names}
            self.assertEqual(third_party, set())

        def test_no_user_interface_or_io(self):
            source = inspect.getsource(can_build_software_camera)
            for forbidden in ("input(", "print(", "open(", "tkinter", "argparse"):
                self.assertNotIn(forbidden, source)

        def test_function_takes_required_ranges_and_camera_list(self):
            params = list(inspect.signature(can_build_software_camera).parameters)
            self.assertEqual(params, ["required_distance", "required_light", "cameras"])
            self.assertIsInstance(ok((0, 1), (0, 1), [(0, 1, 0, 1)]), bool)

    def _reference_quadratic(req_d, req_l, cams):
        """Previous O(n^2) algorithm, kept ONLY as an independent cross-check."""
        d_lo, d_hi = req_d
        l_lo, l_hi = req_l
        rel = [
            Camera(*c) for c in cams
            if c[1] >= d_lo and c[0] <= d_hi and c[3] >= l_lo and c[2] <= l_hi
        ]
        if not rel:
            return False
        rel.sort(key=lambda c: c.light_min)

        def light_ok(sel):
            reach = None
            for c in sel:
                if reach is None:
                    if c.light_min > l_lo:
                        return False
                    reach = c.light_max
                else:
                    if c.light_min > reach:
                        return False
                    reach = max(reach, c.light_max)
                if reach >= l_hi:
                    return True
            return False

        pts = sorted({d_lo, d_hi} | {v for c in rel for v in (c.dist_min, c.dist_max) if d_lo < v < d_hi})
        for i, x in enumerate(pts):
            if not light_ok([c for c in rel if c.dist_min <= x <= c.dist_max]):
                return False
            if i + 1 < len(pts):
                nx = pts[i + 1]
                if not light_ok([c for c in rel if c.dist_min <= x and nx <= c.dist_max]):
                    return False
        return True

    class SegmentTree(unittest.TestCase):
        def test_matches_naive_array(self):
            rng = random.Random(3)
            for size in list(range(1, 40)) + [63, 64, 65, 200]:
                tree = _MinCoverTree(size)
                naive = [0] * size
                for _ in range(120):
                    a, b = sorted((rng.randrange(size), rng.randrange(size)))
                    delta = rng.choice([-2, -1, 1, 1, 2])
                    tree.add(a, b, delta)
                    for k in range(a, b + 1):
                        naive[k] += delta
                    self.assertEqual(tree.minimum(), min(naive), (size, a, b))

        def test_padding_never_counts_as_a_cell(self):
            tree = _MinCoverTree(5)  # padded to 8 with +inf
            tree.add(0, 4, 1)
            self.assertEqual(tree.minimum(), 1)

    class OptimizedCrossChecks(unittest.TestCase):
        def test_matches_reference_on_bigger_random_cases(self):
            rng = random.Random(424242)
            seen = {True: 0, False: 0}
            for _ in range(4000):
                size = rng.choice([4, 8, 20])
                pair = lambda: tuple(sorted((rng.randint(0, size), rng.randint(0, size))))
                req_d, req_l = pair(), pair()
                cams = [pair() + pair() for _ in range(rng.randint(0, 25))]
                got = ok(req_d, req_l, cams)
                self.assertEqual(got, _reference_quadratic(req_d, req_l, cams), (req_d, req_l, cams))
                seen[got] += 1
            self.assertGreater(seen[True], 200)
            self.assertGreater(seen[False], 200)

        def test_matches_reference_with_fractions_floats_and_infinity(self):
            rng = random.Random(11)
            pool = [-INF, -3, Fraction(-5, 2), 0, 0.5, Fraction(1, 3), 1, 2.25, 7, INF]
            pool.sort()
            for _ in range(3000):
                def pair():
                    a, b = sorted((rng.choice(pool), rng.choice(pool)))
                    return a, b
                req_d, req_l = pair(), pair()
                cams = [pair() + pair() for _ in range(rng.randint(0, 12))]
                self.assertEqual(
                    ok(req_d, req_l, cams), _reference_quadratic(req_d, req_l, cams),
                    (req_d, req_l, cams),
                )

        def test_random_tilings_are_covered_and_break_when_a_tile_is_removed(self):
            rng = random.Random(5)
            for _ in range(300):
                xs = sorted(rng.sample(range(1, 60), rng.randint(1, 6)))
                ys = sorted(rng.sample(range(1, 60), rng.randint(1, 6)))
                xs, ys = [0] + xs + [60], [0] + ys + [60]
                tiles = [(xs[i], xs[i + 1], ys[j], ys[j + 1])
                         for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
                rng.shuffle(tiles)
                self.assertTrue(ok((0, 60), (0, 60), tiles))
                self.assertFalse(ok((0, 60), (0, 60), tiles[1:]))

        def test_events_at_the_same_distance_are_applied_before_checking(self):
            # One camera ends exactly where another (covering the same light) starts,
            # plus a third overlapping both: transient states must not fail.
            cams = [(0, 5, 0, 10), (5, 10, 0, 10), (4, 6, 0, 10)]
            self.assertTrue(ok((0, 10), (0, 10), cams))

        def test_gap_only_at_the_far_end_and_only_at_the_near_end(self):
            self.assertFalse(ok((0, 10), (0, 10), [(0, 9, 0, 10)]))
            self.assertFalse(ok((0, 10), (0, 10), [(1, 10, 0, 10)]))
            self.assertFalse(ok((0, 10), (0, 10), [(0, 10, 0, 9)]))
            self.assertFalse(ok((0, 10), (0, 10), [(0, 10, 1, 10)]))

        def test_cameras_extending_far_beyond_the_requirement(self):
            cams = [(-INF, 5, -INF, INF), (5, INF, -INF, INF)]
            self.assertTrue(ok((0, 10), (0, 10), cams))
            self.assertFalse(ok((0, 10), (0, 10), [(-INF, 4.9, -INF, INF), (5, INF, -INF, INF)]))

    class Scalability(unittest.TestCase):
        def test_200k_chained_cameras(self):
            n = 200_000
            cams = [(i, i + 2, 0, 10) for i in range(n)]
            start = time.perf_counter()
            self.assertTrue(ok((0, n + 1), (0, 10), cams))
            elapsed = time.perf_counter() - start
            cams = [c for c in cams if not (n // 2 - 1 <= c[0] <= n // 2 + 1)]
            self.assertFalse(ok((0, n + 1), (0, 10), cams))
            self.assertLess(elapsed, 20)

        def test_300x300_tile_grid_90k_cameras(self):
            k = 300
            tiles = [(i, i + 1, j, j + 1) for i in range(k) for j in range(k)]
            start = time.perf_counter()
            self.assertTrue(ok((0, k), (0, k), tiles))
            elapsed = time.perf_counter() - start
            tiles.remove((123, 124, 45, 46))
            self.assertFalse(ok((0, k), (0, k), tiles))
            self.assertLess(elapsed, 30)

        def test_growth_is_near_linearithmic_not_quadratic(self):
            def run(n):
                cams = [(i, i + 2, 0, 10) for i in range(n)]
                t = time.perf_counter()
                self.assertTrue(ok((0, n + 1), (0, 10), cams))
                return time.perf_counter() - t

            run(2000)  # warm-up
            small, big = run(10_000), run(40_000)  # 4x the input
            # Quadratic would be ~16x; allow generous noise, still far below it.
            self.assertLess(big / small, 9)

        def test_much_faster_than_the_quadratic_reference(self):
            n = 3000
            cams = [(i, i + 2, 0, 10) for i in range(n)]
            t = time.perf_counter()
            fast = ok((0, n + 1), (0, 10), cams)
            t_fast = time.perf_counter() - t
            t = time.perf_counter()
            slow = _reference_quadratic((0, n + 1), (0, 10), cams)
            t_slow = time.perf_counter() - t
            self.assertEqual(fast, slow)
            self.assertLess(t_fast * 5, t_slow)

    class ApiContract(unittest.TestCase):
        def test_returns_plain_bool_and_does_not_mutate_arguments(self):
            cams = [[5, 10, 0, 10], [0, 5, 0, 10]]
            rd, rl = [0, 10], [0, 10]
            snap = ([list(c) for c in cams], list(rd), list(rl))
            self.assertIs(ok(rd, rl, cams), True)
            self.assertIs(ok(rd, rl, cams[:1]), False)
            self.assertEqual(([list(c) for c in cams], rd, rl), snap)

        def test_deterministic(self):
            rng = random.Random(1)
            cams = [tuple(sorted((rng.randint(0, 9), rng.randint(0, 9)))) +
                    tuple(sorted((rng.randint(0, 9), rng.randint(0, 9)))) for _ in range(30)]
            results = {ok((0, 9), (0, 9), rng.sample(cams, len(cams))) for _ in range(50)}
            self.assertEqual(len(results), 1)

    unittest.main(verbosity=2)
