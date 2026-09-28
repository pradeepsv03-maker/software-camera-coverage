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

    Decisions (documented because the statement leaves them open):
      * Ranges are CLOSED and CONTINUOUS. Distance 5 belongs to both [0, 5] and
        [5, 10], so those two cameras together cover [0, 10]. But [0, 5] and
        [5.0001, 10] leave a gap, which is a failure.
      * Two cameras that only touch at a corner do not cover the areas around
        that corner (a diagonal touch is not a cover).
      * A range with min == max is legal (a single value).
      * min > max, NaN, non-numbers, or malformed cameras raise an error rather
        than silently returning a wrong answer. +/- infinity is allowed
        (e.g. "works for any light level above 100").
      * Comparisons are exact. Use ints or fractions.Fraction if you need exact
        decimal boundaries (floats such as 0.1 + 0.2 != 0.3 behave as floats do).

ALGORITHM  (coordinate compression + sweep over the distance axis)
    1. Discard cameras that do not touch the required rectangle.
    2. Collect all distinct distance boundaries inside the required distance
       range. They split the distance axis into "elementary pieces": each
       boundary point itself, and each open gap between neighbouring points.
       Inside one piece the set of cameras covering it never changes, so
       checking one representative per piece is exact (no sampling, no epsilon).
    3. For every piece, take the cameras that span it and check with a greedy
       interval merge that their light ranges cover the required light range.
    4. The answer is True iff every piece passes.

    Time  O(n^2)  (up to 2n+1 pieces x O(n) work each, because cameras are
                   sorted by light_min once, so the per-piece merge needs no sort)
    Space O(n)

    (A sweep line with a segment tree gives O(n log n); the O(n^2) version was
    chosen for clarity and provable correctness. It handles thousands of
    cameras in well under a second or two; see the performance test.)

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


def _light_range_covered(cams: List[Camera], light_lo: float, light_hi: float) -> bool:
    """
    True if the closed light ranges of `cams` (sorted by light_min) cover
    [light_lo, light_hi]. Greedy merge; touching closed intervals join up.
    Precondition (guaranteed by the caller): every camera in `cams` reaches
    light_lo or beyond (light_max >= light_lo) and starts at or below light_hi.
    """
    reach = None  # everything in [light_lo, reach] is known to be covered
    for cam in cams:
        if reach is None:
            if cam.light_min > light_lo:
                return False  # nothing covers light_lo itself
            reach = cam.light_max
        else:
            if cam.light_min > reach:
                return False  # gap between `reach` and this camera
            if cam.light_max > reach:
                reach = cam.light_max
        if reach >= light_hi:
            return True
    return False


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

    # Sorted once; filtering below preserves this order.
    relevant.sort(key=lambda c: c.light_min)

    # Distinct distance boundaries inside the required range.
    breaks = {dist_lo, dist_hi}
    for cam in relevant:
        for value in (cam.dist_min, cam.dist_max):
            if dist_lo < value < dist_hi:
                breaks.add(value)
    points = sorted(breaks)

    for i, x in enumerate(points):
        # Piece 1: the single distance x.
        at_x = [c for c in relevant if c.dist_min <= x <= c.dist_max]
        if not _light_range_covered(at_x, light_lo, light_hi):
            return False
        # Piece 2: the open gap between x and the next boundary.
        if i + 1 < len(points):
            nxt = points[i + 1]
            in_gap = [c for c in relevant if c.dist_min <= x and nxt <= c.dist_max]
            if not _light_range_covered(in_gap, light_lo, light_hi):
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

    unittest.main(verbosity=2)
