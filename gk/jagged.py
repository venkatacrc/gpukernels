"""Vectorised jagged <-> padded conversion (no Python loop over users)."""
import torch


def segment_ids(offsets, total):
    """For each of the `total` jagged rows, the index of the user it belongs to."""
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(torch.arange(counts.numel(), device=offsets.device), counts, output_size=total)


def positions(offsets, total):
    seg = segment_ids(offsets, total)
    return torch.arange(total, device=offsets.device) - offsets[:-1].long()[seg], seg


def to_padded(x, offsets, max_len, fill=0):
    pos, seg = positions(offsets, x.shape[0])
    out = x.new_full((offsets.numel() - 1, max_len) + x.shape[1:], fill)
    out[seg, pos] = x
    return out


def to_jagged(x, offsets):
    pos, seg = positions(offsets, int(offsets[-1]))
    return x[seg, pos]


def last_rows(offsets):
    """Index of each user's last token in the jagged tensor."""
    return offsets[1:].long() - 1
