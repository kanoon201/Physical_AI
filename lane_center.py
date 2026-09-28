"""
Lane center from detected cones
=================================
Turns the cone list from cone_detection into ONE number: where the middle
of the gap is, and how far the car is off it.

    from cone_detection import ConeDetector
    from lane_center import LaneFinder

    det = ConeDetector()
    lane = LaneFinder(det.cfg)

    cones, mask, debug = det.detect(frame)
    result = lane.update(cones, frame.shape[1], frame.shape[0])
    result.offset_norm   # -1.0 = gap is far left, +1.0 = far right, 0 = centered

`offset_norm` is what a steering controller consumes. Deliberately no
controller here: no gains, no throttle, nothing that moves. This file only
answers "where is the middle".

The four things that make this harder than (left + right) / 2
--------------------------------------------------------------
1. Which cone is "left" and which is "right"? Splitting at the frame's
   center only works while the car is already centered — i.e. exactly when
   you don't need the answer. Drift far enough and both cones sit on one
   side of the frame center, the split says "two left cones, no right
   one", and the car is told to turn further the wrong way. Two defenses:
   the split happens at the LAST KNOWN LANE CENTER, which follows the car,
   and when everything still lands on one side the cones are re-split at
   the widest gap between them — which is the gap the car drives through.
   The second one is what saves the very first frame, before there is any
   remembered center to split at.

2. Cones at different distances must not be paired. The nearest cone on
   the left and a far-away one on the right average out to a target that
   is off to one side, and it wobbles every time a distant cone appears or
   drops out. Pairs whose depths are too far apart are refused.

3. One side visible is the normal case in a turn, not an error. Offsetting
   by a guessed fraction of the frame is what the old code did, and that
   guess depends on camera height and real cone spacing. Instead, the half
   gap width is REMEMBERED from the last time both sides were visible, so
   the offset is measured rather than assumed.

4. Cone edges shift a few pixels every frame from JPEG noise, so the raw
   midpoint jitters. The output is smoothed; `smoothing` = 0 disables it.
"""

import cv2

# Merged into the detector's config, so everything lives in one JSON.
DEFAULT_LANE_CONFIG = {
    # Fallback for "only one side visible" before the real gap width has
    # ever been measured — as a fraction of frame width. Only used until
    # both cones are seen once, then the measured value takes over.
    "lane_half_width_ratio": 0.30,

    # Two cones count as a pair only if their bottoms are within this
    # fraction of the frame height of each other.
    "max_pair_depth_diff_ratio": 0.25,

    # EMA on the target x. 0 = raw (jittery), 0.9 = very smooth but laggy.
    "target_smoothing": 0.6,

    # When every cone lands on one side of the split, the cones are
    # re-split at the widest gap between them instead — but only if that
    # gap is at least this fraction of the frame width. Below it, they are
    # more likely to be one boundary line than two.
    "min_lane_gap_ratio": 0.15,

    # Frames with nothing detected before the remembered lane center is
    # thrown away and the split falls back to the frame center.
    "lane_memory_frames": 15,

    # Ignore an offset smaller than this — keeps the car from hunting
    # left-right around a target it is already on.
    "center_deadzone_px": 15,
}

# What update() reports in `state`.
STATE_BOTH = "both"      # both sides visible — the trustworthy case
STATE_LEFT = "left"      # only left cones; target inferred from gap width
STATE_RIGHT = "right"
STATE_LOST = "lost"      # nothing detected; target is the last known one


class LaneResult:
    __slots__ = ("target_x", "frame_center", "offset_px", "offset_norm",
                 "state", "left", "right", "half_width_px", "lost_frames")

    def __init__(self, target_x, frame_center, offset_px, offset_norm,
                 state, left, right, half_width_px, lost_frames):
        self.target_x = target_x            # px, where the middle of the gap is
        self.frame_center = frame_center    # px, where the car is pointing
        self.offset_px = offset_px          # target_x - frame_center, deadzoned
        self.offset_norm = offset_norm      # offset_px scaled to -1..+1
        self.state = state
        self.left = left                    # the Cone used, or None
        self.right = right
        self.half_width_px = half_width_px  # measured half gap, or None
        self.lost_frames = lost_frames

    @property
    def have_target(self):
        """False means the target is a stale guess — don't drive on it."""
        return self.state != STATE_LOST

    def __repr__(self):
        return (f"LaneResult(state={self.state}, target_x={self.target_x:.0f}, "
                f"offset_norm={self.offset_norm:+.2f})")


def _depth(cone):
    """How near a cone is, in pixels. The BOTTOM of the box, not the
    centroid: a cone's base sits on the floor, so its bottom edge tracks
    distance. The centroid also moves when the cone is clipped by the
    frame edge, which distance has nothing to do with."""
    x, y, w, h = cone.bbox
    return y + h


class LaneFinder:
    def __init__(self, cfg=None):
        """cfg: the detector's config dict. Missing lane keys are filled in
        from DEFAULT_LANE_CONFIG, so an old cone_config.json still works."""
        self.cfg = dict(DEFAULT_LANE_CONFIG)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULT_LANE_CONFIG})
            # Keep the caller's dict as the live one so trackbars editing it
            # reach us, but make sure the lane keys exist in it.
            for k, v in self.cfg.items():
                cfg.setdefault(k, v)
            self.cfg = cfg
        self.reset()

    def reset(self):
        self._center_x = None       # last known lane center, drives the L/R split
        self._half_width = None     # measured half gap width, in px
        self._smoothed = None
        self._lost_frames = 0

    def update(self, cones, frame_w, frame_h):
        frame_center = frame_w / 2.0
        split_x = self._center_x if self._center_x is not None else frame_center

        left_cones, right_cones = self._split(cones, split_x, frame_w)

        # Nearest on each side. Depth, not size: a big far cone and a small
        # near one are easy to mix up by area alone.
        near_left = max(left_cones, key=_depth) if left_cones else None
        near_right = max(right_cones, key=_depth) if right_cones else None

        # A pair spanning too much depth is not a gap to aim at.
        if near_left and near_right:
            max_diff = frame_h * float(self.cfg["max_pair_depth_diff_ratio"])
            if abs(_depth(near_left) - _depth(near_right)) > max_diff:
                # Keep the nearer one; the other is too far back to trust.
                if _depth(near_left) >= _depth(near_right):
                    near_right = None
                else:
                    near_left = None

        state, target_x = self._target_from(near_left, near_right, frame_w, frame_center)

        if state == STATE_LOST:
            self._lost_frames += 1
            if self._lost_frames >= int(self.cfg["lane_memory_frames"]):
                # The remembered center is stale enough to be misleading —
                # a wrong split is worse than no split.
                self._center_x = None
                self._smoothed = None
        else:
            self._lost_frames = 0

        # Smooth, but only while we have something real to smooth toward.
        if state != STATE_LOST:
            a = float(self.cfg["target_smoothing"])
            if self._smoothed is None or a <= 0:
                self._smoothed = target_x
            else:
                self._smoothed = a * self._smoothed + (1.0 - a) * target_x
            self._center_x = self._smoothed

        out_x = self._smoothed if self._smoothed is not None else frame_center

        offset_px = out_x - frame_center
        if abs(offset_px) < float(self.cfg["center_deadzone_px"]):
            offset_px = 0.0
        offset_norm = max(-1.0, min(1.0, offset_px / frame_center))

        return LaneResult(out_x, frame_center, offset_px, offset_norm, state,
                          near_left, near_right, self._half_width, self._lost_frames)

    def _split(self, cones, split_x, frame_w):
        """Left/right groups. Normally a straight cut at split_x; if that
        puts every cone on one side, fall back to cutting at the widest
        horizontal gap between the cones themselves."""
        left = [c for c in cones if c.cx < split_x]
        right = [c for c in cones if c.cx >= split_x]
        if (left and right) or len(cones) < 2:
            return left, right

        ordered = sorted(cones, key=lambda c: c.cx)
        gaps = [(ordered[i + 1].cx - ordered[i].cx, i) for i in range(len(ordered) - 1)]
        widest, at = max(gaps)

        # A measured gap beats a guessed one: once the real lane width is
        # known, demand at least half of it.
        min_gap = self._half_width if self._half_width is not None             else frame_w * float(self.cfg["min_lane_gap_ratio"])
        if widest < min_gap:
            # Too narrow to be the lane — these are one boundary, not two.
            return left, right

        cut = (ordered[at].cx + ordered[at + 1].cx) / 2.0
        return [c for c in cones if c.cx < cut], [c for c in cones if c.cx >= cut]

    def _target_from(self, near_left, near_right, frame_w, frame_center):
        if near_left and near_right:
            target = (near_left.cx + near_right.cx) / 2.0
            # Measured gap — this is the only place half_width is learned.
            measured = abs(near_right.cx - near_left.cx) / 2.0
            if self._half_width is None:
                self._half_width = measured
            else:
                self._half_width = 0.7 * self._half_width + 0.3 * measured
            return STATE_BOTH, target

        half = self._half_width
        if half is None:
            half = frame_w * float(self.cfg["lane_half_width_ratio"])

        if near_left:
            # Left boundary only: the gap is to its right.
            return STATE_LEFT, near_left.cx + half
        if near_right:
            return STATE_RIGHT, near_right.cx - half
        return STATE_LOST, frame_center


# ============================================================
# DEBUG DRAWING
# ============================================================

STATE_COLORS = {
    STATE_BOTH: (0, 255, 0),
    STATE_LEFT: (0, 200, 255),
    STATE_RIGHT: (0, 200, 255),
    STATE_LOST: (0, 0, 255),
}


def draw_lane(frame, result):
    """Draw the target on a frame (in place). Pass the detector's debug
    frame so the cone boxes stay visible underneath."""
    h, w = frame.shape[:2]
    color = STATE_COLORS[result.state]
    y = int(h * 0.75)

    # Where the car points, vs. where it should point.
    cv2.line(frame, (int(result.frame_center), 0), (int(result.frame_center), h),
             (255, 255, 255), 1)
    cv2.line(frame, (int(result.target_x), int(h * 0.45)),
             (int(result.target_x), h), color, 2)

    if result.left and result.right:
        # The gap being aimed through.
        cv2.line(frame, (result.left.cx, result.left.cy),
                 (result.right.cx, result.right.cy), color, 1)

    cv2.circle(frame, (int(result.target_x), y), 9, color, -1)
    cv2.arrowedLine(frame, (int(result.frame_center), y), (int(result.target_x), y),
                    color, 2, tipLength=0.3)

    txt = f"{result.state}  offset={result.offset_norm:+.2f}"
    if result.state == STATE_LOST:
        txt += f"  ({result.lost_frames} frames)"
    cv2.putText(frame, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
