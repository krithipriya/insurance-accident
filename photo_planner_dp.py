"""
photo_planner_dp.py

Algorithm to plan the minimal set of photos (camera viewpoints) to cover all regions of a car.
- Uses bitmask dynamic programming (exact) when number of regions M is small (<=20 by default).
- Falls back to a greedy set-cover approximation for larger M.

How it works (simplified pipeline):
1. Model the car as a rectangular prism and sample points on its 6 faces (these are the regions to cover).
2. Generate candidate camera viewpoints around the car (azimuths, distances, heights).
3. For each candidate view compute which sample points lie within the view's field of view -> cover masks.
4. Solve set-cover with DP (exact) or greedy (approx).

This is a self-contained demonstration; in a real app you'd compute cover masks using the real vehicle geometry
or depth maps and include occlusion tests (raycasts) against the vehicle mesh.

Run as a script to see a demonstration.
"""

from typing import List, Tuple
import math
import itertools

INF = 10**9

# ---------- geometry helpers ----------

def normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x,y,z = v
    n = math.sqrt(x*x + y*y + z*z)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

# ---------- car region sampling ----------

def sample_car_surface(length: float = 4.5, width: float = 1.8, height: float = 1.4,
                       samples_per_edge: int = 5) -> List[Tuple[float, float, float]]:
    """
    Sample points on the six faces of the car's bounding box.
    Returns a list of 3D points (x,y,z). Car is centered at origin, x forward, y right, z up.

    length: front-to-back extent (m)
    width: left-to-right extent (m)
    height: vertical extent (m)
    samples_per_edge: controls resolution (>=2)
    """
    L = length / 2.0
    W = width / 2.0
    H = height

    # parameter grids from -1..1 for two axes per face
    vals = [i/(samples_per_edge-1) * 2 - 1 for i in range(samples_per_edge)]

    pts = []
    # Front face (x = +L)
    for a in vals:
        for b in vals:
            y = a * W
            z = max(0.0, b * H)  # only upper half on faces where ground exists
            pts.append((L, y, z))
    # Rear face (x = -L)
    for a in vals:
        for b in vals:
            y = a * W
            z = max(0.0, b * H)
            pts.append((-L, y, z))
    # Left face (y = -W)
    for a in vals:
        for b in vals:
            x = a * L
            z = max(0.0, b * H)
            pts.append((x, -W, z))
    # Right face (y = +W)
    for a in vals:
        for b in vals:
            x = a * L
            z = max(0.0, b * H)
            pts.append((x, W, z))
    # Roof (z = H)
    for a in vals:
        for b in vals:
            x = a * L
            y = b * W
            pts.append((x, y, H))
    # Optional: bumper undersides / ground-level points (z small)
    # sample a small ring near ground for bumpers
    for a in vals:
        for b in [-1.0, 1.0]:
            x = a * L
            y = b * W
            pts.append((x, y, 0.05))

    # Remove duplicates (if any) and return
    unique = []
    seen = set()
    for p in pts:
        key = (round(p[0], 4), round(p[1],4), round(p[2],4))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

# ---------- viewpoints generation ----------

def generate_viewpoints(azimuth_steps: int = 8,
                        distances: List[float] = [2.0, 4.0],
                        heights: List[float] = [1.6],
                        look_at: Tuple[float,float,float] = (0.0,0.0,0.6)) -> List[Tuple[Tuple[float,float,float], Tuple[float,float,float]]]:
    """
    Generate candidate camera viewpoints around the car.
    Returns a list of tuples (cam_pos, cam_forward_unit_vector).
    cam_pos is (x,y,z) and forward points toward look_at.
    """
    views = []
    for az in range(azimuth_steps):
        theta = (az / azimuth_steps) * 2 * math.pi
        for d in distances:
            for h in heights:
                cx = math.cos(theta) * d
                cy = math.sin(theta) * d
                cz = h
                cam = (cx, cy, cz)
                forward = normalize(sub(look_at, cam))
                views.append((cam, forward))
    return views

# ---------- coverage computation ----------

def compute_cover_masks(points: List[Tuple[float,float,float]],
                        views: List[Tuple[Tuple[float,float,float], Tuple[float,float,float]]],
                        fov_deg: float = 70.0,
                        max_distance: float = 10.0) -> List[int]:
    """
    For each view compute a bitmask of which points are inside the camera's angular FOV and within max distance.
    Note: This simple method ignores occlusion (no raycasts). For production use compute occlusion against the vehicle mesh.
    """
    half_angle_cos = math.cos(math.radians(fov_deg) / 2.0)
    cover_masks = []
    for cam_pos, forward in views:
        mask = 0
        for i,p in enumerate(points):
            v = sub(p, cam_pos)
            d = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if d < 1e-6 or d > max_distance:
                continue
            v_norm = (v[0]/d, v[1]/d, v[2]/d)
            cosang = dot(forward, v_norm)
            if cosang >= half_angle_cos:
                # point is inside cone
                mask |= (1 << i)
        cover_masks.append(mask)
    return cover_masks

# ---------- dynamic programming exact solver (bitmask) ----------

def bitmask_dp_min_photos(M: int, coverMasks: List[int]) -> Tuple[int, List[int]]:
    """
    Exact solution using DP over masks. Returns (min_photos, list_of_view_indices).
    If impossible to cover all regions with given views returns (INF, []).
    Complexity: O(2^M * V) time, O(2^M) memory.
    """
    full = (1 << M) - 1
    V = len(coverMasks)
    dp = [INF] * (1 << M)
    parent = [(-1, -1)] * (1 << M)
    dp[0] = 0

    # iterate masks; can be improved by BFS over reachable masks for sparsity
    for mask in range(1 << M):
        if dp[mask] >= INF:
            continue
        # try each view
        for i in range(V):
            new_mask = mask | coverMasks[i]
            if dp[new_mask] > dp[mask] + 1:
                dp[new_mask] = dp[mask] + 1
                parent[new_mask] = (mask, i)

    if dp[full] >= INF:
        return INF, []

    # reconstruct sequence
    seq = []
    cur = full
    while cur != 0:
        prev_mask, view_i = parent[cur]
        if prev_mask == -1:
            break
        seq.append(view_i)
        cur = prev_mask
    seq.reverse()
    return dp[full], seq

# ---------- greedy approximation fallback ----------

def greedy_set_cover(M: int, coverMasks: List[int]) -> Tuple[int, List[int]]:
    full = (1 << M) - 1
    uncovered = full
    chosen = []
    V = len(coverMasks)

    # precompute cardinalities if useful
    while uncovered:
        best_i = -1
        best_gain = 0
        for i in range(V):
            gain_mask = coverMasks[i] & uncovered
            # use bin(...).count('1') for compatibility with older Python versions
            gain = bin(gain_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i == -1:
            break
        chosen.append(best_i)
        uncovered &= ~coverMasks[best_i]

    if uncovered != 0:
        return INF, chosen
    return len(chosen), chosen

# ---------- utility to prune redundant views ----------

def prune_views(coverMasks: List[int]) -> List[int]:
    """
    Remove views whose cover is subset of another view (strict subset), and remove duplicate masks.
    Returns list of indices of views to keep (relative to original list).
    """
    V = len(coverMasks)
    keep = [True] * V
    for i in range(V):
        for j in range(V):
            if i == j: continue
            mi = coverMasks[i]
            mj = coverMasks[j]
            if mi == 0:
                keep[i] = False
            elif mi | mj == mj and mi != mj:
                # i is strict subset of j
                keep[i] = False
    return [i for i,k in enumerate(keep) if k]

# ---------- demo / main ----------

def plan_photos_for_car(samples_per_edge: int = 4,
                        azimuth_steps: int = 8,
                        distances: List[float] = [2.0, 4.0],
                        heights: List[float] = [1.6],
                        fov_deg: float = 70.0,
                        dp_max_regions: int = 20,
                        show_plot: bool = False,
                        show_all_views: bool = False,
                        annotate: bool = False,
                        fov_line_length: float = 6.0,
                        output_prefix: str = 'photo_plan') -> None:
    # 1. sample points on car surface
    points = sample_car_surface(samples_per_edge=samples_per_edge)
    M = len(points)
    print(f"Sampled {M} surface points (regions) on car")

    # 2. generate candidate viewpoints
    views = generate_viewpoints(azimuth_steps=azimuth_steps, distances=distances, heights=heights)
    print(f"Generated {len(views)} candidate viewpoints")

    # 3. compute cover masks
    coverMasks = compute_cover_masks(points, views, fov_deg=fov_deg)

    # 4. prune redundant views
    keep_idx = prune_views(coverMasks)
    coverMasks_pruned = [coverMasks[i] for i in keep_idx]
    print(f"Pruned views: {len(coverMasks_pruned)} kept out of {len(coverMasks)}")

    # 5. solve with DP if small M, else greedy
    if M <= dp_max_regions:
        print("Using exact bitmask DP solver")
        min_photos, seq = bitmask_dp_min_photos(M, coverMasks_pruned)
        if min_photos >= INF:
            print("Exact DP: impossible to cover all regions with current candidate views. Falling back to greedy.")
            g_photos, g_seq = greedy_set_cover(M, coverMasks_pruned)
            print(f"Greedy picked {g_photos} views (indices into pruned list): {g_seq}")
            print("Original view indices:", [keep_idx[i] for i in g_seq])
        else:
            print(f"Exact DP found solution with {min_photos} photos. Pruned-view indices: {seq}")
            print("Original view indices:", [keep_idx[i] for i in seq])
    else:
        print("Too many regions for exact DP; using greedy approximation")
        g_photos, g_seq = greedy_set_cover(M, coverMasks_pruned)
        if g_photos >= INF:
            print("Greedy failed to cover all regions")
        else:
            print(f"Greedy picked {g_photos} views (indices into pruned list): {g_seq}")
            print("Original view indices:", [keep_idx[i] for i in g_seq])
            selected_seq = g_seq

    # Optionally, show which points remain uncovered by the greedy solution for debugging
    # Visualization (top-down and side views) saved to PNG when requested
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except Exception:
        plt = None

    if show_plot:
        if plt is None:
            print("matplotlib not available. To generate plots install it: pip install matplotlib")
        else:
            # determine selected sequence (either from exact DP or greedy)
            sel = []
            if M <= dp_max_regions and 'seq' in locals() and seq:
                sel = seq
            else:
                sel = selected_seq if 'selected_seq' in locals() else []

            # map pruned indices back to original views
            selected_original = [keep_idx[i] for i in sel]

            # helper to get covered point indices for a pruned view index
            def covered_point_indices(pruned_i: int):
                mask = coverMasks_pruned[pruned_i]
                idxs = []
                j = 0
                while mask:
                    if mask & 1:
                        idxs.append(j)
                    mask >>= 1
                    j += 1
                return idxs

            # Top-down (x,y)
            fig, ax = plt.subplots(figsize=(8, 8))
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.scatter(xs, ys, c='lightgray', s=30, label='sample points')

            # optionally plot all candidate pruned views faintly
            if show_all_views:
                for ii, pr in enumerate(range(len(coverMasks_pruned))):
                    orig_i = keep_idx[pr]
                    cam_pos, forward = views[orig_i]
                    cx, cy, cz = cam_pos
                    ax.scatter([cx], [cy], c='0.8', marker='.', s=30)

            # plot selected camera positions and lines to covered points
            for k, pr_i in enumerate(sel):
                orig_i = keep_idx[pr_i]
                cam_pos, forward = views[orig_i]
                cx, cy, cz = cam_pos
                ax.scatter([cx], [cy], c=f'C{k}', marker='^', s=120, edgecolors='k', linewidths=0.8, label=f'cam {k}')
                covered = covered_point_indices(pr_i)
                for pi in covered:
                    px, py, pz = points[pi]
                    ax.plot([cx, px], [cy, py], c=f'C{k}', linewidth=0.8)

                # draw FOV rays (project forward to XY plane)
                yaw = math.atan2(forward[1], forward[0])
                half = math.radians(fov_deg) / 2.0
                for ang in (yaw - half, yaw + half):
                    rx = cx + math.cos(ang) * fov_line_length
                    ry = cy + math.sin(ang) * fov_line_length
                    ax.plot([cx, rx], [cy, ry], c=f'C{k}', linestyle='--', linewidth=1.0)

                if annotate:
                    ax.text(cx, cy, f'{keep_idx[pr_i]}', color='black', fontsize=9, weight='bold')

            ax.set_title('Top-down view (x,y)')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.axis('equal')
            ax.legend()
            top_path = f"{output_prefix}_topdown.png"
            fig.savefig(top_path, dpi=150)
            plt.close(fig)

            # Side view (x,z) - collapse y
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            zs = [p[2] for p in points]
            ax2.scatter(xs, zs, c='lightgray', s=30, label='sample points')
            if show_all_views:
                for ii, pr in enumerate(range(len(coverMasks_pruned))):
                    orig_i = keep_idx[pr]
                    cam_pos, forward = views[orig_i]
                    cx, cy, cz = cam_pos
                    ax2.scatter([cx], [cz], c='0.8', marker='.', s=30)

            for k, pr_i in enumerate(sel):
                orig_i = keep_idx[pr_i]
                cam_pos, forward = views[orig_i]
                cx, cy, cz = cam_pos
                ax2.scatter([cx], [cz], c=f'C{k}', marker='^', s=120, edgecolors='k', linewidths=0.8, label=f'cam {k}')
                covered = covered_point_indices(pr_i)
                for pi in covered:
                    px, py, pz = points[pi]
                    ax2.plot([cx, px], [cz, pz], c=f'C{k}', linewidth=0.8)

                if annotate:
                    ax2.text(cx, cz, f'{keep_idx[pr_i]}', color='black', fontsize=9, weight='bold')

            ax2.set_title('Side view (x,z)')
            ax2.set_xlabel('x (m)')
            ax2.set_ylabel('z (m)')
            ax2.axis('equal')
            ax2.legend()
            side_path = f"{output_prefix}_side.png"
            fig2.savefig(side_path, dpi=150)
            plt.close(fig2)

            print(f"Saved visualization images: {top_path}, {side_path}")


def generate_synthetic_damage_image(samples_per_edge: int = 4,
                                    damaged_parts: List[str] = ['front','left'],
                                    output_prefix: str = 'car_damage') -> str:
    """
    Generate a synthetic car schematic (top-down) and highlight damaged parts selected from sampled points.
    damaged_parts: list of strings chosen from ['front','rear','left','right','roof']
    Returns path to saved PNG.
    """
    pts = sample_car_surface(samples_per_edge=samples_per_edge)
    # map part -> predicate on point
    L = max(abs(p[0]) for p in pts) if pts else 1.0
    W = max(abs(p[1]) for p in pts) if pts else 1.0
    H = max(p[2] for p in pts) if pts else 1.0

    predicates = {
        'front': lambda p: p[0] > 0.6 * L,
        'rear': lambda p: p[0] < -0.6 * L,
        'left': lambda p: p[1] < -0.4 * W,
        'right': lambda p: p[1] > 0.4 * W,
        'roof': lambda p: p[2] > 0.8 * H,
    }

    damaged_idx = set()
    for part in damaged_parts:
        part = part.strip().lower()
        if part in predicates:
            for i,p in enumerate(pts):
                if predicates[part](p):
                    damaged_idx.add(i)

    damaged_pts = [pts[i] for i in sorted(damaged_idx)]

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception:
        raise RuntimeError('matplotlib required to generate synthetic damage image')

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, c='lightgray', s=40)

    if damaged_pts:
        dxs = [p[0] for p in damaged_pts]
        dys = [p[1] for p in damaged_pts]
        ax.scatter(dxs, dys, c='red', s=80, label='damaged')

        # polygon hull via angle sort around centroid
        cx = sum(dxs)/len(dxs)
        cy = sum(dys)/len(dys)
        angs = [math.atan2(y-cy, x-cx) for x,y in zip(dxs, dys)]
        pts_sorted_xy = [(p[0], p[1]) for _,p in sorted(zip(angs, damaged_pts))]
        poly = Polygon(pts_sorted_xy, closed=True, facecolor=(1,0,0,0.25), edgecolor='red')
        ax.add_patch(poly)

    ax.set_title('Synthetic car top-down view with damaged regions')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.axis('equal')
    ax.legend()
    out_path = f"{output_prefix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path

if __name__ == '__main__':
    # small example; adjust samples_per_edge to increase/decrease number of regions
    # CLI options: --show-plot, --all-views, --annotate, --fov-length, --output-prefix
    import argparse

    parser = argparse.ArgumentParser(description='Photo planner demo with optional visualization')
    parser.add_argument('--samples', type=int, default=3, help='samples_per_edge')
    parser.add_argument('--azimuth-steps', type=int, default=8)
    parser.add_argument('--distances', type=float, nargs='+', default=[2.5, 4.5])
    parser.add_argument('--heights', type=float, nargs='+', default=[1.6])
    parser.add_argument('--fov', type=float, default=70.0)
    parser.add_argument('--show-plot', action='store_true')
    parser.add_argument('--all-views', action='store_true', help='Also plot all candidate pruned views faintly')
    parser.add_argument('--annotate', action='store_true', help='Annotate selected camera indices on plots')
    parser.add_argument('--fov-length', type=float, default=6.0, help='Length of FOV lines in plots')
    parser.add_argument('--output-prefix', type=str, default='photo_plan')
    parser.add_argument('--synthetic-damage', type=str, default='', help='Comma-separated parts: front,rear,left,right,roof')

    args = parser.parse_args()

    plan_photos_for_car(samples_per_edge=args.samples,
                        azimuth_steps=args.azimuth_steps,
                        distances=args.distances,
                        heights=args.heights,
                        fov_deg=args.fov,
                        dp_max_regions=20,
                        show_plot=args.show_plot,
                        show_all_views=args.all_views,
                        annotate=args.annotate,
                        fov_line_length=args.fov_length,
                        output_prefix=args.output_prefix)

    if args.synthetic_damage:
        parts = [p.strip() for p in args.synthetic_damage.split(',') if p.strip()]
        try:
            out = generate_synthetic_damage_image(samples_per_edge=args.samples, damaged_parts=parts, output_prefix=args.output_prefix + '_damage')
            print(f"Saved synthetic damage image: {out}")
        except Exception as e:
            print(f"Failed to generate synthetic damage image: {e}")
