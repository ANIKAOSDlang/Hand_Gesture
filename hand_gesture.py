"""
ASL (American Sign Language) Hand Gesture Recognition - A to Z
==============================================================
Educational tool: ওয়েবক্যামে ASL fingerspelling signs দেখিয়ে
A থেকে Z পর্যন্ত সকল ইংরেজি alphabet শিখুন।

Uses MediaPipe HandLandmarker to detect 21 hand landmarks and
classifies the hand pose into one of 26 ASL letters.

Controls:
  q - Quit
  r - Toggle reference guide panel
  s - Toggle settings panel
  b - Toggle auto low-light boost
  [ / ]   - Decrease / Increase manual brightness
  1..7    - Select background color preset (when settings panel open)

ASL Fingerspelling Signs:
  A - Fist, thumb beside index finger
  B - Four fingers up straight, thumb folded across palm
  C - Curved hand forming "C" shape
  D - Index up, other fingers touch thumb tip
  E - All fingers curled, thumb tucked in front
  F - Index+thumb form circle, other 3 fingers up
  G - Fist, index+thumb point sideways
  H - Index+middle point sideways together
  I - Fist, pinky up
  J - Pinky up and moving (shown as I with motion)
  K - Index+middle up spread, thumb between them
  L - L-shape: index up + thumb out
  M - Fist, thumb under 3 fingers
  N - Fist, thumb under 2 fingers
  O - All fingertips touch thumb forming "O"
  P - K-sign pointing downward
  Q - G-sign pointing downward
  R - Index+middle crossed/together up
  S - Fist, thumb over fingers
  T - Fist, thumb between index+middle
  U - Index+middle up together
  V - Index+middle up spread (peace sign)
  W - Index+middle+ring up spread
  X - Index finger hooked/bent
  Y - Thumb+pinky out, others closed (hang loose)
  Z - Index draws "Z" in air (shown as index pointing)
"""

import os
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision import HandLandmarksConnections

# ── MediaPipe Hand Landmark IDs ──────────────────────────────────
# Fingertip IDs
THUMB_TIP = 4;  THUMB_IP = 3;  THUMB_MCP = 2;  THUMB_CMC = 1
INDEX_TIP = 8;  INDEX_DIP = 7; INDEX_PIP = 6;   INDEX_MCP = 5
MIDDLE_TIP = 12; MIDDLE_DIP = 11; MIDDLE_PIP = 10; MIDDLE_MCP = 9
RING_TIP = 16;  RING_DIP = 15; RING_PIP = 14;   RING_MCP = 13
PINKY_TIP = 20; PINKY_DIP = 19; PINKY_PIP = 18;  PINKY_MCP = 17
WRIST = 0

TIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
DIP_IDS = [THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]
PIP_IDS = [THUMB_MCP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
MCP_IDS = [THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# ASL reference descriptions for UI
ASL_DESCRIPTIONS = {
    'A': "Fist, thumb beside",
    'B': "4 fingers up, thumb in",
    'C': "Curved C shape",
    'D': "Index up, rest on thumb",
    'E': "Fingers curled down",
    'F': "OK sign, 3 fingers up",
    'G': "Index+thumb sideways",
    'H': "Index+mid sideways",
    'I': "Fist, pinky up",
    'J': "Pinky up (like I)",
    'K': "Index+mid up, spread",
    'L': "L shape: index+thumb",
    'M': "Fist, thumb under 3",
    'N': "Fist, thumb under 2",
    'O': "Fingertips touch thumb",
    'P': "K pointing down",
    'Q': "G pointing down",
    'R': "Index+mid up together",
    'S': "Fist, thumb over",
    'T': "Thumb between idx+mid",
    'U': "Index+mid up together",
    'V': "Peace/victory sign",
    'W': "3 fingers up spread",
    'X': "Index finger hooked",
    'Y': "Thumb+pinky out",
    'Z': "Index pointing out",
}


# ══════════════════════════════════════════════════════════════════
# Helper geometry functions
# ══════════════════════════════════════════════════════════════════

def dist(a, b):
    """Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def dist_2d(a, b):
    """2D Euclidean distance (x, y only)."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def angle_3pts(a, b, c):
    """Angle at point b formed by segments ba and bc (in degrees)."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    mag_ba = math.sqrt(sum(v**2 for v in ba))
    mag_bc = math.sqrt(sum(v**2 for v in bc))
    if mag_ba * mag_bc == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def is_finger_up(lm, finger_idx, handedness):
    """Check if a finger is extended (up/straight).
    finger_idx: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    """
    if finger_idx == 0:  # Thumb
        if handedness == "Right":
            return lm[THUMB_TIP].x < lm[THUMB_IP].x
        else:
            return lm[THUMB_TIP].x > lm[THUMB_IP].x
    tip = TIP_IDS[finger_idx]
    pip = PIP_IDS[finger_idx]
    return lm[tip].y < lm[pip].y


def is_finger_curled(lm, finger_idx):
    """Check if a finger is curled (bent significantly)."""
    if finger_idx == 0:
        return dist(lm[THUMB_TIP], lm[THUMB_MCP]) < dist(lm[THUMB_MCP], lm[THUMB_CMC]) * 0.7
    tip = TIP_IDS[finger_idx]
    dip = DIP_IDS[finger_idx]
    pip = PIP_IDS[finger_idx]
    mcp = MCP_IDS[finger_idx]
    # Finger is curled when tip is below PIP or very close to MCP
    return lm[tip].y > lm[pip].y or dist(lm[tip], lm[mcp]) < dist(lm[pip], lm[mcp]) * 0.6


def is_finger_hooked(lm, finger_idx):
    """Check if a finger is hooked (DIP bent but MCP/PIP somewhat extended)."""
    if finger_idx == 0:
        return False
    tip = TIP_IDS[finger_idx]
    dip = DIP_IDS[finger_idx]
    pip = PIP_IDS[finger_idx]
    # Hooked: tip is below DIP but DIP is above or near PIP
    return lm[tip].y > lm[dip].y and lm[dip].y < lm[pip].y


def get_finger_states(lm, handedness):
    """Return list of 5 booleans: which fingers are up."""
    return [is_finger_up(lm, i, handedness) for i in range(5)]


def thumb_is_across_palm(lm, handedness):
    """Check if thumb is folded across the palm (for B sign)."""
    # Thumb tip is between index and middle MCP horizontally
    thumb_x = lm[THUMB_TIP].x
    idx_mcp_x = lm[INDEX_MCP].x
    mid_mcp_x = lm[MIDDLE_MCP].x
    # Thumb tip should be near the palm center
    palm_center_x = (lm[INDEX_MCP].x + lm[PINKY_MCP].x) / 2
    return abs(thumb_x - palm_center_x) < abs(idx_mcp_x - mid_mcp_x) * 2


def thumb_beside_fist(lm, handedness):
    """Check if thumb is beside the fist (not over, not tucked) - for A."""
    # Thumb should be roughly at the level of index PIP/MCP
    thumb_y = lm[THUMB_TIP].y
    idx_pip_y = lm[INDEX_PIP].y
    idx_mcp_y = lm[INDEX_MCP].y
    return idx_mcp_y - 0.05 < thumb_y < idx_pip_y + 0.05


def fingertips_touching_thumb(lm):
    """Check if fingertips are touching/near the thumb tip (for O, E)."""
    threshold = 0.06
    touching = 0
    for tip_id in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        if dist(lm[tip_id], lm[THUMB_TIP]) < threshold:
            touching += 1
    return touching


def hand_pointing_sideways(lm, handedness):
    """Check if hand is pointing sideways (for G, H)."""
    # Index finger should have significant x-direction extension
    dx = abs(lm[INDEX_TIP].x - lm[INDEX_MCP].x)
    dy = abs(lm[INDEX_TIP].y - lm[INDEX_MCP].y)
    return dx > dy * 1.2


def hand_pointing_down(lm):
    """Check if hand/fingers point downward (for P, Q)."""
    # Middle finger tip or index tip is below wrist
    return lm[MIDDLE_TIP].y > lm[WRIST].y or lm[INDEX_TIP].y > lm[MIDDLE_MCP].y


def thumb_index_circle(lm):
    """Check if thumb and index form a circle/touching (for F, O, D)."""
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.05


def index_middle_crossed(lm):
    """Check if index and middle fingers are crossed (for R)."""
    # When crossed, index tip is closer to middle DIP side and vice versa
    idx_mid_tip_dist = dist_2d(lm[INDEX_TIP], lm[MIDDLE_TIP])
    idx_mid_mcp_dist = dist_2d(lm[INDEX_MCP], lm[MIDDLE_MCP])
    return idx_mid_tip_dist < idx_mid_mcp_dist * 0.5


# ══════════════════════════════════════════════════════════════════
# Main ASL letter classifier
# ══════════════════════════════════════════════════════════════════

def classify_asl_letter(lm, handedness):
    """
    Classify the hand pose into an ASL letter (A-Z).
    Each letter has a UNIQUE hand sign based on ASL fingerspelling.
    lm = list of 21 landmarks
    handedness = "Left" or "Right"
    Returns: (letter string A-Z, confidence description) or "?" if uncertain
    """
    fingers = get_finger_states(lm, handedness)
    thumb_up, index_up, middle_up, ring_up, pinky_up = fingers
    num_fingers_up = sum(fingers)

    # Count curled fingers
    curled = [is_finger_curled(lm, i) for i in range(5)]

    # Distance metrics
    thumb_index_dist = dist(lm[THUMB_TIP], lm[INDEX_TIP])
    thumb_middle_dist = dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
    thumb_ring_dist = dist(lm[THUMB_TIP], lm[RING_TIP])

    palm_size = dist(lm[WRIST], lm[MIDDLE_MCP])
    sideways = hand_pointing_sideways(lm, handedness)
    pointing_down = hand_pointing_down(lm)

    # Normalized threshold for "touching"
    close = palm_size * 0.3

    # Index-middle tip distance (used for U vs V vs R)
    idx_mid_tip = dist_2d(lm[INDEX_TIP], lm[MIDDLE_TIP])

    # ─── GROUP 1: Unique easy-to-detect signs first ──────────

    # Y: ONLY thumb + pinky out, 3 middle fingers closed
    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "Y"

    # I: ONLY pinky up (fist + pinky)  |  J = I with motion
    if pinky_up and not index_up and not middle_up and not ring_up and not thumb_up:
        return "I"

    # ─── GROUP 2: Sideways pointing signs ────────────────────

    # G: Index+thumb point SIDEWAYS, others closed, hand NOT pointing down
    if sideways and not pointing_down and index_up and not middle_up and not ring_up and not pinky_up:
        return "G"

    # H: Index+middle point SIDEWAYS together, others closed
    if sideways and not pointing_down and index_up and middle_up and not ring_up and not pinky_up:
        return "H"

    # ─── GROUP 3: Downward pointing signs ────────────────────

    # Q: Like G but pointing DOWN (index+thumb down)
    if pointing_down and not middle_up and not ring_up and not pinky_up and thumb_index_dist < close * 2:
        return "Q"

    # P: Like K but pointing DOWN (index+middle down)
    if pointing_down and index_up and middle_up and not ring_up and not pinky_up:
        return "P"

    # ─── GROUP 4: L-shape ────────────────────────────────────

    # L: Thumb out + index up = L shape, others closed
    if thumb_up and index_up and not middle_up and not ring_up and not pinky_up and not sideways:
        return "L"

    # ─── GROUP 5: Thumb+index touching + other fingers up ────

    # F: Thumb+index touch (circle), middle+ring+pinky UP
    if thumb_index_dist < close and middle_up and ring_up and pinky_up:
        return "F"

    # ─── GROUP 6: Two fingers up variations ──────────────────

    # D: ONLY index up, other 3 tips near/touching thumb
    if index_up and not middle_up and not ring_up and not pinky_up and not sideways:
        if thumb_middle_dist < close or thumb_ring_dist < close:
            return "D"

    # X: Index HOOKED (bent at DIP), others closed
    if is_finger_hooked(lm, 1) and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "X"

    # K: Index+middle up SPREAD + thumb UP between them
    if index_up and middle_up and not ring_up and not pinky_up and thumb_up and not sideways:
        return "K"

    # R: Index+middle up and CROSSED (tips very close)
    if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        if index_middle_crossed(lm):
            return "R"

    # U: Index+middle up TOGETHER (parallel, close tips), thumb NOT up
    if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        if idx_mid_tip < palm_size * 0.22:
            return "U"

    # V: Index+middle up SPREAD (peace sign), thumb NOT up
    if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        if idx_mid_tip >= palm_size * 0.22:
            return "V"

    # ─── GROUP 7: Three fingers up ───────────────────────────

    # W: Index+middle+ring UP spread, pinky closed, thumb closed
    if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
        return "W"

    # ─── GROUP 8: Four/five fingers up ───────────────────────

    # B: Four fingers up straight, thumb folded in
    if index_up and middle_up and ring_up and pinky_up and not thumb_up:
        return "B"

    # B (variant): All 5 fingers up = open hand
    if all(fingers):
        return "B"

    # ─── GROUP 9: All fingers down - O, C, E shapes ─────────

    # O: Fingertips touch/near thumb tip (circle)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        touching = fingertips_touching_thumb(lm)
        if touching >= 2:
            return "O"

    # C: Curved hand (partially bent fingers, gap between thumb & index)
    if num_fingers_up <= 1 and not any(curled[1:]):
        index_angle = angle_3pts(lm[INDEX_MCP], lm[INDEX_PIP], lm[INDEX_TIP])
        if 60 < index_angle < 150:
            if dist(lm[THUMB_TIP], lm[INDEX_TIP]) > close:
                return "C"

    # E: All 4 fingers curled down, thumb in front
    if not index_up and not middle_up and not ring_up and not pinky_up:
        if all(curled[1:]) and lm[THUMB_TIP].y < lm[INDEX_PIP].y:
            return "E"

    # ─── GROUP 10: Fist variations (A, S, T, M, N) ──────────
    if num_fingers_up <= 1 and not index_up and not middle_up and not ring_up and not pinky_up:
        thumb_y = lm[THUMB_TIP].y
        idx_pip_y = lm[INDEX_PIP].y
        mid_pip_y = lm[MIDDLE_PIP].y
        ring_pip_y = lm[RING_PIP].y

        # T: Thumb pokes BETWEEN index and middle knuckles
        thumb_near_idx = dist(lm[THUMB_TIP], lm[INDEX_PIP]) < close
        thumb_near_mid = dist(lm[THUMB_TIP], lm[MIDDLE_PIP]) < close
        if thumb_near_idx and thumb_near_mid and thumb_y < idx_pip_y:
            return "T"

        # M: Thumb tucked UNDER 3 fingers (below index+middle+ring PIPs)
        if thumb_y > idx_pip_y and thumb_y > mid_pip_y and thumb_y > ring_pip_y:
            return "M"

        # N: Thumb tucked UNDER 2 fingers (below index+middle PIPs only)
        if thumb_y > idx_pip_y and thumb_y > mid_pip_y and thumb_y <= ring_pip_y:
            return "N"

        # S: Fist with thumb OVER fingers (thumb in front, closer to camera)
        if lm[THUMB_TIP].z < lm[INDEX_PIP].z:
            return "S"

        # A: Fist with thumb BESIDE index (default fist)
        return "A"

    # ─── GROUP 11: Single finger up (fallback) ───────────────

    # Z: Only index up pointing (Z is drawn in air with index)
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Z"

    return "?"


# ══════════════════════════════════════════════════════════════════
# Settings & low-light helpers
# ══════════════════════════════════════════════════════════════════

# Background color presets (BGR). The tint is blended over dark areas
# of the frame to make the picture more pleasant in low light.
BG_PRESETS = [
    ("Off",         None),
    ("Warm Amber",  (40, 110, 200)),    # cozy orange-amber
    ("Soft Cyan",   (200, 180, 80)),    # cool studio cyan
    ("Mint",        (140, 210, 130)),   # soft mint green
    ("Lavender",    (220, 150, 170)),   # purple/lavender
    ("Rose",        (150, 130, 230)),   # warm pink
    ("Neutral Gray",(120, 120, 120)),   # neutral lift
]


class Settings:
    """User-tunable runtime settings."""
    def __init__(self):
        self.show_settings = False
        self.show_reference = True
        self.bg_index = 0              # index into BG_PRESETS
        self.bg_strength = 0.35        # 0.0 - 0.9 blend strength
        self.auto_low_light = True     # auto brightness when dark
        self.manual_gain = 1.0         # extra multiplicative gain (0.5 - 2.5)

    @property
    def bg_name(self):
        return BG_PRESETS[self.bg_index][0]

    @property
    def bg_color(self):
        return BG_PRESETS[self.bg_index][1]

    def cycle_bg(self, step=1):
        self.bg_index = (self.bg_index + step) % len(BG_PRESETS)

    def select_bg(self, idx):
        if 0 <= idx < len(BG_PRESETS):
            self.bg_index = idx

    def adjust_gain(self, delta):
        self.manual_gain = max(0.5, min(2.5, self.manual_gain + delta))

    def adjust_strength(self, delta):
        self.bg_strength = max(0.0, min(0.9, self.bg_strength + delta))


def estimate_brightness(frame):
    """Mean luminance (0-255) of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def apply_low_light_enhancement(frame, settings, brightness):
    """Brighten dark frames + apply selected background color tint.

    Returns the (possibly modified) frame.
    """
    out = frame

    # 1) Auto brightness boost when scene is dark.
    auto_gain = 1.0
    if settings.auto_low_light and brightness < 90:
        # Smooth ramp: at 30 lux -> 1.8x, at 90 lux -> 1.0x
        auto_gain = 1.0 + (90 - brightness) / 75.0
        auto_gain = max(1.0, min(2.0, auto_gain))

    gain = auto_gain * settings.manual_gain
    if abs(gain - 1.0) > 0.01:
        out = cv2.convertScaleAbs(out, alpha=gain, beta=8)

    # 2) Background color tint (soft overlay weighted toward dark areas).
    color = settings.bg_color
    if color is not None and settings.bg_strength > 0.01:
        h, w = out.shape[:2]
        tint = np.full((h, w, 3), color, dtype=np.uint8)

        # Mask: stronger tint where the frame is darker (helps low light
        # without washing out a well-lit hand).
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        # darkness: 1.0 at black, 0.0 at white
        darkness = 1.0 - (gray.astype(np.float32) / 255.0)
        mask = (darkness * settings.bg_strength)[:, :, None]
        out = (out.astype(np.float32) * (1.0 - mask)
               + tint.astype(np.float32) * mask).astype(np.uint8)

    return out


def draw_rounded_rect(img, pt1, pt2, color, alpha=0.85, radius=12, border=None):
    """Draw a semi-transparent rounded rectangle in-place."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    # Body
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    # Corners
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if border is not None:
        c, t = border
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), c, t)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), c, t)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), c, t)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), c, t)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, c, t)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, c, t)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, c, t)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, c, t)


# ══════════════════════════════════════════════════════════════════
# Drawing utilities
# ══════════════════════════════════════════════════════════════════

def draw_hand_landmarks(frame, landmarks, connections):
    """Draw hand landmarks and connections on the frame."""
    h, w, _ = frame.shape
    pts = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        pts.append((px, py))
        cv2.circle(frame, (px, py), 5, (0, 255, 0), cv2.FILLED)

    for conn in connections:
        start, end = conn
        if start < len(pts) and end < len(pts):
            cv2.line(frame, pts[start], pts[end], (255, 255, 255), 2)


# Short sign instructions for each letter (shown in reference panel)
SIGN_INSTRUCTIONS = {
    'A': "Fist+thumb side",
    'B': "4 fingers up",
    'C': "Curved C shape",
    'D': "Index up only",
    'E': "Curl all down",
    'F': "OK+3 fingers up",
    'G': "Point sideways",
    'H': "2 fingers side",
    'I': "Pinky up only",
    'J': "Pinky up(=I)",
    'K': "2 up+thumb mid",
    'L': "L: idx+thumb",
    'M': "Fist thumb<3",
    'N': "Fist thumb<2",
    'O': "Tips touch O",
    'P': "K point down",
    'Q': "G point down",
    'R': "2 fingers cross",
    'S': "Fist thumb over",
    'T': "Thumb between",
    'U': "2 up together",
    'V': "Peace sign",
    'W': "3 fingers up",
    'X': "Index hooked",
    'Y': "Thumb+pinky",
    'Z': "Index point(=1)",
}


def draw_reference_panel(frame, current_letter):
    """Draw a responsive ASL alphabet reference panel on the right side."""
    h, w, _ = frame.shape
    scale = h / 480.0  # Base scale factor (480p as reference)

    panel_w = int(240 * scale)
    panel_w = min(panel_w, w // 3)  # Never exceed 1/3 of screen
    panel_x = w - panel_w

    # Rounded translucent panel
    draw_rounded_rect(
        frame, (panel_x + 4, 8), (w - 6, h - 8),
        color=(28, 28, 38), alpha=0.78, radius=14,
        border=((0, 200, 255), 1),
    )

    # Title
    title_size = max(0.45, 0.55 * scale)
    title_thick = max(1, int(1.5 * scale))
    cv2.putText(frame, "ASL Sign Guide", (panel_x + int(16 * scale), int(28 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, title_size, (0, 220, 255), title_thick)
    line_y = int(38 * scale)
    cv2.line(frame, (panel_x + 12, line_y), (w - 14, line_y), (80, 80, 110), 1)

    # Each letter with its sign description
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    start_y = line_y + int(18 * scale)
    available_h = h - start_y - int(10 * scale)
    row_h = max(int(14 * scale), available_h // 26)
    font_letter = max(0.35, 0.4 * scale)
    font_desc = max(0.28, 0.32 * scale)

    for i, letter in enumerate(letters):
        ly = start_y + i * row_h
        if ly > h - 10:
            break

        is_current = (letter == current_letter)
        letter_color = (0, 255, 0) if is_current else (200, 200, 200)
        desc_color = (100, 255, 100) if is_current else (140, 140, 140)
        thickness = max(1, int(1.5 * scale)) if is_current else 1

        # Highlight background for current letter
        if is_current:
            cv2.rectangle(frame, (panel_x + 3, ly - int(12 * scale)), (w - 3, ly + int(4 * scale)),
                          (0, 80, 0), cv2.FILLED)

        # Letter
        cv2.putText(frame, f"{letter}:", (panel_x + int(8 * scale), ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_letter, letter_color, thickness)

        # Sign description
        desc = SIGN_INSTRUCTIONS.get(letter, "")
        cv2.putText(frame, desc, (panel_x + int(30 * scale), ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_desc, desc_color, 1)


def draw_status_panel(frame, letter, description, finger_states):
    """Draw a responsive info panel at top-left."""
    h, w, _ = frame.shape
    scale = h / 480.0  # Base scale factor

    # Panel dimensions (responsive)
    panel_w = int(420 * scale)
    panel_w = min(panel_w, w // 2)  # Never exceed half screen
    panel_h = int(190 * scale)
    panel_h = min(panel_h, h // 3)

    # Rounded panel with subtle border
    draw_rounded_rect(
        frame, (8, 8), (panel_w, panel_h),
        color=(35, 35, 50), alpha=0.82, radius=14,
        border=((0, 220, 220), 2),
    )

    # Title
    title_size = max(0.5, 0.6 * scale)
    title_thick = max(1, int(1.5 * scale))
    cv2.putText(frame, "ASL Alphabet Detector", (int(20 * scale), int(34 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, title_size, (0, 240, 255), title_thick)

    # Detected letter
    letter_size = max(0.7, 0.9 * scale)
    letter_thick = max(1, int(2 * scale))
    cv2.putText(frame, f"Letter: {letter}", (int(20 * scale), int(72 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, letter_size, (80, 255, 120), letter_thick)

    # Description
    if description:
        desc_size = max(0.4, 0.5 * scale)
        cv2.putText(frame, f"Sign: {description}", (int(20 * scale), int(102 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, desc_size, (210, 210, 220), 1)

    # Finger states — full names
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_y = int(132 * scale)
    dot_r = max(6, int(9 * scale))
    name_size = max(0.3, 0.38 * scale)
    spacing = max(60, int(panel_w / 5.5))

    for i, (name, state) in enumerate(zip(finger_names, finger_states)):
        color = (80, 255, 120) if state else (80, 80, 220)
        x = int(20 * scale) + i * spacing
        # Dot
        cv2.circle(frame, (x + dot_r, finger_y), dot_r, color, cv2.FILLED)
        cv2.circle(frame, (x + dot_r, finger_y), dot_r, (255, 255, 255), 1)
        # Full name below dot
        label = "UP" if state else "--"
        cv2.putText(frame, name, (x, finger_y + dot_r + int(14 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, name_size, (200, 200, 200), 1)
        cv2.putText(frame, label, (x + 2, finger_y + dot_r + int(28 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, name_size, color, 1)


def draw_settings_panel(frame, settings, brightness):
    """Center settings panel showing background color presets and toggles."""
    h, w, _ = frame.shape
    scale = h / 480.0

    pw = min(int(440 * scale), w - 40)
    ph = min(int(360 * scale), h - 40)
    x1 = (w - pw) // 2
    y1 = (h - ph) // 2
    x2 = x1 + pw
    y2 = y1 + ph

    draw_rounded_rect(
        frame, (x1, y1), (x2, y2),
        color=(25, 25, 40), alpha=0.92, radius=18,
        border=((0, 220, 255), 2),
    )

    pad = int(18 * scale)
    cv2.putText(frame, "Settings", (x1 + pad, y1 + int(34 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.6, 0.75 * scale),
                (0, 230, 255), max(1, int(2 * scale)))
    cv2.line(frame, (x1 + pad, y1 + int(44 * scale)),
             (x2 - pad, y1 + int(44 * scale)), (80, 80, 110), 1)

    # Background presets
    cv2.putText(frame, "Background color (low light):",
                (x1 + pad, y1 + int(70 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.5 * scale),
                (220, 220, 230), 1)

    swatch_y = y1 + int(82 * scale)
    sw = int(36 * scale)
    sh = int(36 * scale)
    gap = int(8 * scale)
    label_size = max(0.32, 0.4 * scale)

    for i, (name, color) in enumerate(BG_PRESETS):
        sx = x1 + pad + i * (sw + gap)
        sy = swatch_y
        if color is None:
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (60, 60, 70), cv2.FILLED)
            cv2.line(frame, (sx, sy), (sx + sw, sy + sh), (200, 80, 80), 2)
        else:
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, cv2.FILLED)
        # Selected outline
        if i == settings.bg_index:
            cv2.rectangle(frame, (sx - 2, sy - 2), (sx + sw + 2, sy + sh + 2),
                          (0, 255, 255), 2)
        # Hotkey number
        cv2.putText(frame, str(i + 1), (sx + 4, sy + sh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, label_size, (255, 255, 255), 1)

    cv2.putText(frame, f"Selected: {settings.bg_name}",
                (x1 + pad, swatch_y + sh + int(22 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.38, 0.45 * scale),
                (180, 240, 255), 1)

    # Strength bar
    bar_y = swatch_y + sh + int(40 * scale)
    cv2.putText(frame, f"Tint Strength: {int(settings.bg_strength * 100)}%   (-/+ keys)",
                (x1 + pad, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.38, 0.45 * scale),
                (220, 220, 230), 1)
    bw = pw - 2 * pad
    bx = x1 + pad
    by = bar_y + int(8 * scale)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + int(8 * scale)), (60, 60, 75), cv2.FILLED)
    fill = int(bw * (settings.bg_strength / 0.9))
    cv2.rectangle(frame, (bx, by), (bx + fill, by + int(8 * scale)), (0, 220, 255), cv2.FILLED)

    # Brightness controls
    info_y = by + int(34 * scale)
    auto_label = "ON" if settings.auto_low_light else "OFF"
    auto_color = (80, 255, 120) if settings.auto_low_light else (120, 120, 140)
    cv2.putText(frame, f"Auto low-light boost: {auto_label}   (b)",
                (x1 + pad, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.38, 0.45 * scale),
                auto_color, 1)

    cv2.putText(frame,
                f"Manual gain: x{settings.manual_gain:.2f}   ([ / ])",
                (x1 + pad, info_y + int(22 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.38, 0.45 * scale),
                (220, 220, 230), 1)

    cv2.putText(frame, f"Scene brightness: {brightness:.0f} / 255",
                (x1 + pad, info_y + int(44 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.38, 0.45 * scale),
                (180, 200, 220), 1)

    # Hint footer
    cv2.putText(frame,
                "Keys: 1-7 color | -/+ strength | b auto | [/] gain | s close",
                (x1 + pad, y2 - int(16 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.32, 0.38 * scale),
                (160, 160, 180), 1)


# ══════════════════════════════════════════════════════════════════
# Smoothing: avoids flickering by requiring consistent detection
# ══════════════════════════════════════════════════════════════════

class LetterSmoother:
    """Smooths the detected letter over multiple frames."""
    def __init__(self, buffer_size=7):
        self.buffer = []
        self.buffer_size = buffer_size
        self.current = "?"

    def update(self, letter):
        self.buffer.append(letter)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Find most common letter in buffer
        if self.buffer:
            from collections import Counter
            counts = Counter(self.buffer)
            most_common = counts.most_common(1)[0]
            # Require at least ~40% agreement
            if most_common[1] >= max(2, self.buffer_size * 0.4):
                self.current = most_common[0]
        return self.current


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    # Path to model file (next to this script)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Download it from: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    # Create HandLandmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Build connections list as (int, int) tuples
    hand_connections = [
        (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
    ]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("=" * 55)
    print("  ASL Hand Gesture Recognition - A to Z")
    print("  Educational Fingerspelling Detector")
    print("=" * 55)
    print("  Controls:")
    print("    q       - Quit")
    print("    r       - Toggle reference guide panel")
    print("    s       - Toggle settings panel")
    print("    b       - Toggle auto low-light boost")
    print("    [ / ]   - Manual brightness gain down/up")
    print("    - / +   - Decrease/Increase tint strength")
    print("    1..7    - Select background color preset")
    print("=" * 55)

    smoother = LetterSmoother(buffer_size=7)
    settings = Settings()
    collected_letters = ""

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Flip for selfie-view
        frame = cv2.flip(frame, 1)

        # Apply low-light enhancement & background color tint BEFORE
        # detection so the model sees the same brightened image.
        brightness = estimate_brightness(frame)
        frame = apply_low_light_enhancement(frame, settings, brightness)
        h, w, _ = frame.shape

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        raw_letter = "?"
        finger_states = [False] * 5

        if results.hand_landmarks and results.handedness:
            for hand_lms, hand_info in zip(results.hand_landmarks, results.handedness):
                # Draw hand skeleton
                draw_hand_landmarks(frame, hand_lms, hand_connections)

                # Determine handedness
                label = hand_info[0].category_name  # "Left" or "Right"

                # Get finger states for display
                finger_states = get_finger_states(hand_lms, label)

                # Classify the ASL letter
                raw_letter = classify_asl_letter(hand_lms, label)

        # Smooth the detection
        display_letter = smoother.update(raw_letter)
        description = ASL_DESCRIPTIONS.get(display_letter, "")

        # ── Draw UI elements ─────────────────────────────────
        # Reference panel (right side)
        if settings.show_reference:
            draw_reference_panel(frame, display_letter)

        # Status panel (top-left)
        draw_status_panel(frame, display_letter, description, finger_states)

        # ── Show large letter in the center (responsive) ────
        if display_letter and display_letter != "?":
            scale = h / 480.0
            big_font = max(3, 4.5 * scale)
            big_thick = max(5, int(7 * scale))
            shadow_thick = max(7, int(9 * scale))

            text_size = cv2.getTextSize(display_letter, cv2.FONT_HERSHEY_SIMPLEX, big_font, big_thick)[0]
            # Center horizontally, accounting for reference panel
            available_w = w
            if settings.show_reference:
                panel_w = int(240 * scale)
                panel_w = min(panel_w, w // 3)
                available_w = w - panel_w
            text_x = (available_w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2

            # Shadow
            cv2.putText(frame, display_letter, (text_x + 3, text_y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, big_font, (0, 0, 0), shadow_thick)
            # Letter
            cv2.putText(frame, display_letter, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, big_font, (255, 255, 255), big_thick)

        # ── Bottom bar (responsive) ──────────────────────────
        scale = h / 480.0
        bar_h = max(30, int(40 * scale))
        bar_font = max(0.35, 0.45 * scale)
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (22, 22, 30), cv2.FILLED)
        bg_indicator = settings.bg_name if settings.bg_color else "Off"
        boost = "on" if settings.auto_low_light else "off"
        cv2.putText(
            frame,
            f"r:Ref  s:Settings  b:Boost({boost})  BG:{bg_indicator}  q:Quit",
            (int(12 * scale), h - int(bar_h * 0.32)),
            cv2.FONT_HERSHEY_SIMPLEX, bar_font,
            (170, 170, 190), 1,
        )

        # Settings panel on top of everything
        if settings.show_settings:
            draw_settings_panel(frame, settings, brightness)

        cv2.imshow("ASL Hand Gesture Recognition - A to Z", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            break
        elif key == ord("r"):
            settings.show_reference = not settings.show_reference
        elif key == ord("s"):
            settings.show_settings = not settings.show_settings
        elif key == ord("b"):
            settings.auto_low_light = not settings.auto_low_light
        elif key == ord("["):
            settings.adjust_gain(-0.1)
        elif key == ord("]"):
            settings.adjust_gain(+0.1)
        elif key in (ord("-"), ord("_")):
            settings.adjust_strength(-0.05)
        elif key in (ord("="), ord("+")):
            settings.adjust_strength(+0.05)
        elif key == ord("n"):
            settings.cycle_bg(+1)
        elif ord("1") <= key <= ord("9"):
            settings.select_bg(key - ord("1"))

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
