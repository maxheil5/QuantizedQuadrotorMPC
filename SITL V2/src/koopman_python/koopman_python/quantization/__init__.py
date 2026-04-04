"""SciTech-style quantization modules for the V2 learned-controller path."""

from koopman_python.quantization.dither import dither_signal
from koopman_python.quantization.partition import WordLengthPartition, partition_word_length, quantize_with_partition

__all__ = [
    "WordLengthPartition",
    "dither_signal",
    "partition_word_length",
    "quantize_with_partition",
]
