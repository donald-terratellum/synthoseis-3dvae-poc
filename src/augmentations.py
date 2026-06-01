import numpy as np


DEFAULT_VERTICAL_WARP_MIN_STEP = 0.8
DEFAULT_VERTICAL_WARP_MAX_STEP = 1.25
DEFAULT_VERTICAL_WARP_EDGE_PAD = 5.0
DEFAULT_VERTICAL_WARP_Z_LOW_RATIO = 12.5 / 15.5
DEFAULT_VERTICAL_WARP_Z_MODE_RATIO = 1.0
DEFAULT_VERTICAL_WARP_Z_HIGH_RATIO = 18.5 / 15.5


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
