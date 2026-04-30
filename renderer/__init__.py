"""
renderer — Track A: MuseTalk face-generation engine for edlio-presence

Public exports:
  MuseTalkEngine   — main engine class
  AudioFeatures    — audio input dataclass (lingua franca with Track C)
  RendererFrame    — output frame dataclass
"""

from renderer.engine import AudioFeatures, MuseTalkEngine, RendererFrame

__all__ = ["MuseTalkEngine", "AudioFeatures", "RendererFrame"]
