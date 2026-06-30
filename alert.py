# alert.py — sustained-risk alarm and optional clip recording

import cv2
import datetime
import os
from config import (ALERT_RISK_LEVELS, ALERT_SUSTAIN_FRAMES,
                    RECORD_ON_ALERT, DISPLAY_W, DISPLAY_H)
from logger import save_snapshot


class AlertManager:
    """
    Tracks how many consecutive frames the risk has been HIGH/CRITICAL.
    Fires snapshot + optional video recording when threshold is crossed.
    """

    def __init__(self):
        self._sustain       = 0
        self._alerted       = False
        self._writer        = None
        self._writer_path   = None

    def update(self, risk, frame, frame_idx, annotated_frame):
        """
        Call once per frame.
        Returns True if currently in alert state.
        """
        in_alert_zone = risk in ALERT_RISK_LEVELS

        if in_alert_zone:
            self._sustain += 1
        else:
            self._sustain = 0
            self._alerted = False
            self._stop_recording()

        # Fire alert on first crossing
        if self._sustain >= ALERT_SUSTAIN_FRAMES and not self._alerted:
            self._alerted = True
            snap = save_snapshot(frame, risk, frame_idx)
            print(f"[ALERT] {risk} sustained — snapshot: {snap}")
            if RECORD_ON_ALERT:
                self._start_recording()

        # Feed frames into clip
        if self._writer and self._alerted:
            self._writer.write(annotated_frame)

        return self._alerted

    def _start_recording(self):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"alert_clip_{ts}.avi"
        fourcc       = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(path, fourcc, 20,
                                       (DISPLAY_W, DISPLAY_H))
        self._writer_path = path
        print(f"[RECORD] Saving clip → {path}")

    def _stop_recording(self):
        if self._writer:
            self._writer.release()
            print(f"[RECORD] Clip saved → {self._writer_path}")
            self._writer = self._writer_path = None

    def release(self):
        self._stop_recording()

    @property
    def is_active(self):
        return self._alerted