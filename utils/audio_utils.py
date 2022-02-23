import audioread

"""
# Useful functions for loading audio files
"""

def get_audio_length(path):
    with audioread.audio_open(path) as f:
        return f.duration
