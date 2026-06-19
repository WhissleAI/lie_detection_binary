"""Binary deception detector on the Real-life Trial Deception dataset.

Pipeline: Whissle STT (text + metadata) + audio-visual hybrid intelligence
(MediaPipe emotion/pose/gaze/gesture) + prosody -> multimodal feature matrix
-> speaker-independent classification of deceptive vs. truthful.
"""

__version__ = "0.1.0"
