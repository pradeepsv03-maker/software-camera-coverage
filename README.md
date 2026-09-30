# Software camera coverage check

Decides whether a set of hardware cameras can together support a required range of
subject distances and light levels. **Python 3.10+, standard library only, single file.**

```python
from software_camera import Camera, can_build_software_camera

can_build_software_camera(
    required_distance=(1, 10),   # (min, max)
    required_light=(0, 100),     # (min, max)
    cameras=[Camera(1, 5, 0, 100), Camera(5, 10, 0, 100)],
)  # True
```
Cameras are `Camera(dist_min, dist_max, light_min, light_max)` or plain 4-tuples.

Run the tests: `python software_camera.py`

## Approach: O(n log n) sweep line + segment tree
Each camera is a closed rectangle in the (distance, light) plane; the question is whether the
union of rectangles contains the required rectangle.

1. Validate input, drop cameras that do not touch the required rectangle.
2. **Coordinate compression with doubling** on both axes: each boundary gets an even index `2k`,
   each open gap between boundaries an odd index `2k+1`. Every elementary piece (a point or a gap)
   is one integer cell, and a closed range `[a, b]` is exactly cells `2*idx(a) .. 2*idx(b)`.
   No epsilon, no sampling, no special cases for touching edges.
3. Sweep over distance cells. A camera adds +1 on its light-cell range when the sweep reaches it
   and -1 right after its last cell. A bottom-up segment tree (range add, global minimum) tracks
   the minimum coverage over all light cells.
4. After each batch of events the coverage is constant until the next batch, so checking
   `minimum > 0` there is exact. Any zero means an uncovered cell -> `False`.

Time **O(n log n)**, space **O(n)** (previous version: O(n^2)).

| cameras | previous O(n^2) | this version |
|---:|---:|---:|
| 2,000 | 0.28 s | 0.01 s |
| 5,000 | 1.68 s | 0.02 s |
| 100,000 | (minutes) | 0.59 s |
| 400,000 | - | 1.9 s |

(chain of overlapping cameras, one machine; your numbers will differ)

## Decisions on open points
- Ranges are closed and continuous: [0, 5] and [5, 10] cover [0, 10]; [0, 5] and [5.0001, 10] do not.
- Cameras touching only at a corner do not cover the area around it.
- `min == max` is valid. Infinite bounds are allowed.
- `min > max`, NaN, non-numbers or malformed cameras raise `ValueError` / `TypeError`
  (all cameras are validated, even irrelevant ones).
- Comparisons are exact; use ints or `Fraction` for exact decimal boundaries.

## Assignment requirements
| Requirement | Status |
|---|---|
| Function takes required distance/light ranges and a camera list, returns whether they suffice | `can_build_software_camera(...)` returns a `bool` |
| Any language | Python |
| Analytical problem, no UI | no input/output/UI code (tested) |
| No third-party libraries | imports checked by a test |
| Single file if possible | `software_camera.py`, tests included |
| Code actually runs | compiles, 59 tests pass |

## Verification performed
- `python -m py_compile` clean; `mypy --strict` reports 0 issues in the solution code.
- 59 tests, **100% line coverage**.
- Scenario tests: touching/overlapping/gapped cameras, tiles, donut hole, diagonal touch, empty list, realistic example.
- Independent oracles: brute-force doubled-grid checker and the previous O(n^2) implementation,
  over ~13,000 random cases (integers, floats, `Fraction`s, +/-infinity), plus scale/shift/reorder invariance and random tilings.
- Segment tree checked against a naive array over many sizes (including power-of-two boundaries).
- Scale: 200k chained cameras and a 300x300 tile grid (90k cameras) finish in seconds; a test asserts growth is near-linearithmic.
- Mutation check: 18 deliberate bugs; 16 caught, the other 2 are equivalent mutants that cannot change any result.
