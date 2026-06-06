import numpy as np


DEFAULT_VERTICAL_WARP_MIN_STEP = 0.5
DEFAULT_VERTICAL_WARP_MAX_STEP = 2.0
DEFAULT_VERTICAL_WARP_EDGE_PAD = 5.0
DEFAULT_VERTICAL_WARP_Z_LOW_RATIO = 12.5 / 15.5
DEFAULT_VERTICAL_WARP_Z_MODE_RATIO = 1.0
DEFAULT_VERTICAL_WARP_Z_HIGH_RATIO = 18.5 / 15.5
DEFAULT_MIXUP_SCALE_LOW = 1.0 / 150.0
DEFAULT_MIXUP_SCALE_MODE = 1.0 / 110.0
DEFAULT_MIXUP_SCALE_HIGH = 1.0 / 75.0


def sample_vertical_warp_target_indices(
    nz,
    min_step=DEFAULT_VERTICAL_WARP_MIN_STEP,
    max_step=DEFAULT_VERTICAL_WARP_MAX_STEP,
    edge_pad=DEFAULT_VERTICAL_WARP_EDGE_PAD,
    z_low_ratio=DEFAULT_VERTICAL_WARP_Z_LOW_RATIO,
    z_mode_ratio=DEFAULT_VERTICAL_WARP_Z_MODE_RATIO,
    z_high_ratio=DEFAULT_VERTICAL_WARP_Z_HIGH_RATIO,
    max_tries=32,
):
    if nz < 2:
        return np.arange(nz, dtype=np.float32)

    z_max = float(nz - 1)
    center_idx = 0.5 * z_max

    # Keep local increments in [min_step, max_step] while preserving endpoints.
    feasible_low = max(min_step * center_idx, z_max - max_step * (z_max - center_idx), 0.0)
    feasible_high = min(max_step * center_idx, z_max - min_step * (z_max - center_idx), z_max)
    if feasible_low > feasible_high:
        raise ValueError('No feasible vertical warp midpoint for the provided step constraints.')

    low = np.clip(z_low_ratio * center_idx, feasible_low, feasible_high)
    mode = np.clip(z_mode_ratio * center_idx, feasible_low, feasible_high)
    high = np.clip(z_high_ratio * center_idx, feasible_low, feasible_high)
    low, mode, high = sorted((float(low), float(mode), float(high)))

    current_indices = np.arange(nz, dtype=np.float32)
    control_src = np.array(
        [-edge_pad, 0.0, center_idx, z_max, z_max + edge_pad],
        dtype=np.float32,
    )

    for _ in range(max_tries):
        z_rand = float(np.random.triangular(low, mode, high))
        control_dst = np.array(
            [-edge_pad, 0.0, z_rand, z_max, z_max + edge_pad],
            dtype=np.float32,
        )
        target_indices = np.interp(current_indices, control_src, control_dst).astype(np.float32)
        increments = np.diff(target_indices)
        if increments.size == 0:
            return target_indices
        if float(increments.min()) >= min_step and float(increments.max()) <= max_step:
            return target_indices

    # Fallback to identity if constraints are unexpectedly not met.
    return current_indices


def apply_vertical_warp_to_cube(cube, target_indices):
    nz = cube.shape[-1]
    if nz != int(target_indices.shape[0]):
        raise ValueError('target_indices length must match cube depth.')

    source_indices = np.arange(nz, dtype=np.float32)
    flat_in = cube.reshape(-1, nz)
    flat_out = np.empty_like(flat_in, dtype=np.float32)
    for trace_idx in range(flat_in.shape[0]):
        flat_out[trace_idx] = np.interp(target_indices, source_indices, flat_in[trace_idx]).astype(np.float32)
    return flat_out.reshape(cube.shape)


def keep_trace_extrema_only(x):
    # Keep only local peak/trough samples along each trace (z axis).
    if x.shape[-1] < 3:
        return np.zeros_like(x)

    left = x[:, :, :-2]
    mid = x[:, :, 1:-1]
    right = x[:, :, 2:]
    extrema_mask = ((mid > left) & (mid > right)) | ((mid < left) & (mid < right))

    out = np.zeros_like(x)
    out[:, :, 1:-1][extrema_mask] = x[:, :, 1:-1][extrema_mask]
    return out


def sample_sparse_keep_fraction(fraction_min, fraction_max):
    if float(fraction_min) == float(fraction_max):
        return np.float32(fraction_min)
    return np.float32(np.random.uniform(float(fraction_min), float(fraction_max)))


def _flat_to_xyz(flat_idx, ny, nz):
    x = int(flat_idx // (ny * nz))
    rem = int(flat_idx % (ny * nz))
    y = int(rem // nz)
    z = int(rem % nz)
    return x, y, z


def _build_sparse_indices_poisson_like(shape, target_count, radius_scale):
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    nvox = int(nx * ny * nz)
    if target_count >= nvox:
        return np.arange(nvox, dtype=np.int64)

    radius = float(radius_scale) * float(np.cbrt(float(nvox) / float(target_count)))
    radius2 = radius * radius
    cell_size = max(radius, 1e-6)
    max_checks = min(nvox, max(4096, 8 * int(target_count)))

    buckets = {}
    selected = []
    selected_mask = np.zeros(nvox, dtype=bool)
    candidate_stream = np.random.permutation(nvox)
    checks = 0

    for flat_idx in candidate_stream:
        if checks >= max_checks or len(selected) >= target_count:
            break
        checks += 1

        cx, cy, cz = _flat_to_xyz(int(flat_idx), ny, nz)
        bx = int(np.floor(cx / cell_size))
        by = int(np.floor(cy / cell_size))
        bz = int(np.floor(cz / cell_size))

        keep = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (bx + dx, by + dy, bz + dz)
                    if key not in buckets:
                        continue
                    for ox, oy, oz in buckets[key]:
                        dist2 = float((cx - ox) ** 2 + (cy - oy) ** 2 + (cz - oz) ** 2)
                        if dist2 < radius2:
                            keep = False
                            break
                    if not keep:
                        break
                if not keep:
                    break
            if not keep:
                break

        if not keep:
            continue

        selected.append(int(flat_idx))
        selected_mask[int(flat_idx)] = True
        key = (bx, by, bz)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append((cx, cy, cz))

    if len(selected) < target_count:
        remaining = np.flatnonzero(~selected_mask)
        np.random.shuffle(remaining)
        needed = target_count - len(selected)
        selected.extend(int(v) for v in remaining[:needed])

    return np.asarray(selected[:target_count], dtype=np.int64)


def _build_sparse_indices_uniform_threshold(shape, keep_fraction):
    probs = np.random.uniform(0.0, 1.0, tuple(int(v) for v in shape))
    return np.flatnonzero((probs <= float(keep_fraction)).reshape(-1)).astype(np.int64, copy=False)


def apply_input_random_sparse_keep(
    x,
    fraction_min=0.10,
    fraction_max=0.30,
    method='random',
    poisson_radius_scale=0.85,
):
    nx, ny, nz = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    nvox = int(nx * ny * nz)
    keep_fraction = float(sample_sparse_keep_fraction(fraction_min, fraction_max))
    target_count = int(np.clip(np.rint(keep_fraction * nvox), 1, nvox))

    selected_method = method
    if selected_method == 'random':
        selected_method = 'poisson' if np.random.random() < 0.5 else 'uniform'
    if selected_method == 'poisson':
        selected = _build_sparse_indices_poisson_like((nx, ny, nz), target_count, poisson_radius_scale)
    elif selected_method == 'uniform':
        selected = _build_sparse_indices_uniform_threshold((nx, ny, nz), keep_fraction)
    else:
        raise ValueError("method must be one of: 'random', 'poisson', 'uniform'.")

    # Pair augmentations can produce non-contiguous views (swap/flip). Use contiguous
    # buffers so flat indexing writes into the real output array, not a temporary copy.
    flat_in = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(x.shape, dtype=np.float32)
    flat_out = out.reshape(-1)
    flat_out[selected] = flat_in[selected]
    return out


def _parity_indices(length, parity):
    idx = np.arange(int(parity), int(length), 2, dtype=np.int64)
    if idx.size == 0:
        return np.array([0], dtype=np.int64)
    return idx


def apply_input_decimate_trilinear(x, parity=None):
    nx, ny, nz = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    p = int(np.random.randint(0, 2)) if parity is None else int(parity)
    if p not in (0, 1):
        raise ValueError('parity must be 0 or 1 when provided.')

    idx_x = _parity_indices(nx, p)
    idx_y = _parity_indices(ny, p)
    idx_z = _parity_indices(nz, p)

    anchors = x[np.ix_(idx_x, idx_y, idx_z)].astype(np.float32, copy=False)
    full_z = np.arange(nz, dtype=np.float32)
    full_y = np.arange(ny, dtype=np.float32)
    full_x = np.arange(nx, dtype=np.float32)
    anchor_z = idx_z.astype(np.float32)
    anchor_y = idx_y.astype(np.float32)
    anchor_x = idx_x.astype(np.float32)

    interp_z = np.empty((idx_x.size, idx_y.size, nz), dtype=np.float32)
    for ix in range(idx_x.size):
        for iy in range(idx_y.size):
            interp_z[ix, iy, :] = np.interp(
                full_z,
                anchor_z,
                anchors[ix, iy, :],
                left=float(anchors[ix, iy, 0]),
                right=float(anchors[ix, iy, -1]),
            ).astype(np.float32)

    interp_y = np.empty((idx_x.size, ny, nz), dtype=np.float32)
    for ix in range(idx_x.size):
        for iz in range(nz):
            interp_y[ix, :, iz] = np.interp(
                full_y,
                anchor_y,
                interp_z[ix, :, iz],
                left=float(interp_z[ix, 0, iz]),
                right=float(interp_z[ix, -1, iz]),
            ).astype(np.float32)

    out = np.empty((nx, ny, nz), dtype=np.float32)
    for iy in range(ny):
        for iz in range(nz):
            out[:, iy, iz] = np.interp(
                full_x,
                anchor_x,
                interp_y[:, iy, iz],
                left=float(interp_y[0, iy, iz]),
                right=float(interp_y[-1, iy, iz]),
            ).astype(np.float32)

    out[np.ix_(idx_x, idx_y, idx_z)] = x[np.ix_(idx_x, idx_y, idx_z)]
    return out


def apply_pair_augmentations(x, y, swap_xy_prob, flip_x_prob, flip_y_prob, vertical_warp_prob):
    # Geometric transforms are applied to both input and target.
    if np.random.random() < swap_xy_prob:
        x = np.swapaxes(x, 0, 1)
        y = np.swapaxes(y, 0, 1)
    if np.random.random() < flip_x_prob:
        x = x[::-1, :, :]
        y = y[::-1, :, :]
    if np.random.random() < flip_y_prob:
        x = x[:, ::-1, :]
        y = y[:, ::-1, :]

    # Non-linear depth warp is applied to the clean label first, then mirrored to input.
    if np.random.random() < vertical_warp_prob:
        target_indices = sample_vertical_warp_target_indices(y.shape[-1])
        y = apply_vertical_warp_to_cube(y, target_indices)
        x = y.copy()
    return x, y


def sample_mixup_corpus_index(current_idx, num_examples):
    if num_examples <= 1:
        return int(current_idx)

    mixup_idx = int(np.random.randint(0, int(num_examples) - 1))
    if mixup_idx >= int(current_idx):
        mixup_idx += 1
    return mixup_idx


def apply_input_extrema_mixup(
    x,
    mixup_source,
    scale_low=DEFAULT_MIXUP_SCALE_LOW,
    scale_mode=DEFAULT_MIXUP_SCALE_MODE,
    scale_high=DEFAULT_MIXUP_SCALE_HIGH,
):
    # Keep only peak/trough amplitudes in the secondary example before blending.
    mixup_extrema = keep_trace_extrema_only(mixup_source)
    mixup_scale = np.float32(np.random.triangular(scale_low, scale_mode, scale_high))
    return (x + mixup_scale * mixup_extrema).astype(np.float32, copy=False)


def apply_input_trace_dropout(x, zero_cluster_min, zero_cluster_max):
    # Zero 3x3 XY trace clusters through all Z samples on input only.
    nx, ny, _ = x.shape
    if nx < 3 or ny < 3 or zero_cluster_max == 0:
        return x

    n_clusters = np.random.randint(zero_cluster_min, zero_cluster_max + 1)
    center_x = np.arange(1, nx - 1)
    center_y = np.arange(1, ny - 1)
    max_centers = int(center_x.size * center_y.size)
    if max_centers == 0:
        return x
    n_clusters = min(n_clusters, max_centers)

    flat_choices = np.random.choice(max_centers, size=n_clusters, replace=False)
    for idx in flat_choices:
        cx = int(idx // center_y.size) + 1
        cy = int(idx % center_y.size) + 1
        x[cx - 1:cx + 2, cy - 1:cy + 2, :] = 0.0
    return x
