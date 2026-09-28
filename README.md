# Software camera coverage check

Decides whether a set of hardware cameras can together support a required range of
subject distances and light levels. Python 3.10+, standard library only, single file.

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

## Approach
Each camera is a closed rectangle in the (distance, light) plane, so the question is whether
the union of rectangles contains the required rectangle.

1. Drop cameras that do not touch the required rectangle.
2. Split the distance axis at every camera boundary into pieces: each boundary point and each open gap between neighbours. Inside a piece the set of covering cameras never changes, so one check per piece is exact (no sampling, no epsilon).
3. For each piece, greedily merge the light ranges of the cameras spanning it and check they cover the required light range.
4. Answer `True` only if every piece passes.

Time O(n²), space O(n). (A sweep line with a segment tree would give O(n log n); I chose the simpler, easily verified version.)

## Decisions on open points
- Ranges are closed and continuous: [0, 5] and [5, 10] cover [0, 10]; [0, 5] and [5.0001, 10] do not.
- Cameras touching only at a corner do not cover the area around it.
- min == max is a valid range. Infinite bounds are allowed.
- min > max, NaN, non-numbers or malformed cameras raise `ValueError`/`TypeError`.
- Comparisons are exact; use ints or `Fraction` for exact decimal boundaries.

## Assignment requirements
| Requirement | Status |
|---|---|
| Function takes required distance/light ranges and a camera list, returns whether they suffice | `can_build_software_camera(...)` returns a `bool` |
| Any language | Python |
| Analytical problem, no UI | no input, output or UI code (tested) |
| No third-party libraries | imports checked by a test |
| Single file if possible | `software_camera.py`, tests included |
| Code actually runs | 44 tests pass, 100% line coverage |

## Testing
- Scenario tests: touching, overlapping and gapped cameras, tiles, a donut hole, diagonal touch, empty list, and a realistic example.
- 6,000 random cases compared with an independent brute-force checker, plus invariance under scaling, shifting and reordering.
- Edge cases: point and line requirements, zero-width cameras, infinities, fractions, float rounding, invalid input.
- Performance: 2,000 cameras and a 40x40 tile grid run in about a second.
- Mutation check: 15 deliberate bugs, all caught by the tests.
