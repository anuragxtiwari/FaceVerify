"""
Patches onnxruntime.InferenceSession to default to memory-conservative
settings (disabled memory arena, single-threaded) unless the caller
explicitly passes sess_options.

Why this exists: onnxruntime's default CPU memory arena pre-allocates
growing chunks of memory per model session, often far more than the model
actually needs. On a memory-capped host (like Render's 512MB free tier),
this is a common cause of OOM even for small models. InsightFace's own
model-loading code (FaceAnalysis / model_zoo.get_model) doesn't expose a
way to pass custom session options through — so we patch the constructor
itself, once, before any model is loaded.

This has NO effect on detection or embedding correctness — only on how
much memory onnxruntime reserves internally.

Import this (for its side effect) BEFORE creating any InsightFace models.
"""

import onnxruntime as ort

_original_init = ort.InferenceSession.__init__
_patched = False


def _low_memory_init(self, path_or_bytes, sess_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _original_init(self, path_or_bytes, sess_options=sess_options, **kwargs)


if not _patched:
    ort.InferenceSession.__init__ = _low_memory_init
    _patched = True