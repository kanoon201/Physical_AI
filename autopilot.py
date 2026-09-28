"""
Autopilot: lane offset -> steer and throttle
==============================================
The only file that decides how the car moves. Takes the LaneResult from
lane_center and returns two numbers in -1..+1, ready for RobotLink.drive.

    ap = Autopilot(det.cfg)
    steer, throttle, reason = ap.update(lane_result, dt)

`reason` is a short string for the HUD — always know WHY the car is doing
what it is doing, especially when it stops.

Design notes
------------
P only, no I, no D. Integral term on a steering loop with this much
latency (WiFi + JPEG + decode, easily 150 ms) winds up and oscillates,
and D amplifies the per-frame jitter that lane_center is already
smoothing away. If it hunts, lower the gain before adding terms.

Three things that are not in a textbook P controller but matter here:

1. SLOW DOWN IN TURNS. Throttle is scaled down as |steer| grows. A car
   that takes corners at straight-line speed understeers wide, loses the
   inside cone, and then has nothing to steer by.

2. RAMP, DON'T STEP. Throttle rises toward its target at a limited rate.
   Stepping from 0 to full on a DC motor lurches the car and blurs the
   next few frames, exactly when the detector needs them most.
   The ramp does NOT apply on the way down — braking is immediate.

3. FAIL STOPPED. Losing the lane for `stop_after_lost_frames` means
   stop, not "carry on with the last steering angle". A stale angle in a
   turn drives a curve into whatever is beside the track.
"""

DEFAULT_AUTOPILOT_CONFIG = {
    # steer = offset_norm * steer_gain, clamped to +-max_steer.
    # Start low. Too much gain looks like it is working on a straight and
    # then oscillates violently on the first correction.
    "steer_gain": 0.8,
    "max_steer": 1.0,

    # Speed on a clear straight, -1..1. 0.35 is a sensible crawl for a
    # first run; the loop is easier to debug slower than it is fast.
    "base_throttle": 0.35,

    # Throttle multiplier at full lock. 0.5 = half speed in the tightest
    # turn, scaled linearly with |steer| in between.
    "turn_slowdown": 0.5,

    # Max throttle increase per second while ramping up.
    "throttle_ramp_per_s": 1.0,

    # Lane lost for this many consecutive frames -> stop.
    "stop_after_lost_frames": 5,

    # While only one side is visible the target is inferred, not measured,
    # so back off the speed a little.
    "one_side_throttle_scale": 0.7,
}

REASON_DRIVING = "driving"
REASON_ONE_SIDE = "one side"
REASON_COASTING = "lane lost - coasting"   # inside the grace period, still moving
REASON_SEARCHING = "lane lost - stopped"
REASON_DISARMED = "disarmed"


class Autopilot:
    def __init__(self, cfg=None):
        """cfg: the shared config dict (the detector's). Lane and autopilot
        settings all live in the same cone_config.json."""
        self.cfg = dict(DEFAULT_AUTOPILOT_CONFIG)
        if cfg is not None:
            for k, v in self.cfg.items():
                cfg.setdefault(k, v)
            self.cfg = cfg
        self.reset()

    def reset(self):
        self._throttle = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.last_reason = REASON_DISARMED

    def update(self, lane, dt):
        """lane: a LaneResult. dt: seconds since the previous call.
        Returns (steer, throttle, reason)."""
        from lane_center import STATE_BOTH, STATE_LOST

        # --- steering ---
        steer = lane.offset_norm * float(self.cfg["steer_gain"])
        max_steer = float(self.cfg["max_steer"])
        steer = max(-max_steer, min(max_steer, steer))

        # --- how fast do we want to be going? ---
        if lane.state == STATE_LOST and lane.lost_frames >= int(self.cfg["stop_after_lost_frames"]):
            # Hard stop, and drop the steering too: holding the last angle
            # while blind is how a car leaves the track.
            self._throttle = 0.0
            self.last_steer, self.last_throttle = 0.0, 0.0
            self.last_reason = REASON_SEARCHING
            return 0.0, 0.0, REASON_SEARCHING

        target = float(self.cfg["base_throttle"])

        # Slower the harder we are turning.
        turn_scale = 1.0 - (1.0 - float(self.cfg["turn_slowdown"])) * (abs(steer) / max_steer)
        target *= turn_scale

        if lane.state != STATE_BOTH:
            target *= float(self.cfg["one_side_throttle_scale"])
            # Lost but still inside the grace window: the car is MOVING, so
            # don't label it "stopped" — the HUD would be lying at the one
            # moment it matters.
            reason = REASON_ONE_SIDE if lane.state != STATE_LOST else REASON_COASTING
        else:
            reason = REASON_DRIVING

        # --- ramp up, but brake instantly ---
        if target > self._throttle:
            step = float(self.cfg["throttle_ramp_per_s"]) * max(0.0, dt)
            self._throttle = min(target, self._throttle + step)
        else:
            self._throttle = target

        self.last_steer, self.last_throttle, self.last_reason = steer, self._throttle, reason
        return steer, self._throttle, reason
