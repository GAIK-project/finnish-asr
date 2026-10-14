# QAD-1 transcripts of the Ajokortti clip

These 29 files are GAIK's earlier runs of the same clip (QAD-1, GAIK's re-run of the University of
Helsinki team's evaluation of this clip), copied in their original `ArmResult` format from GAIK's
internal evaluation repository (commit d5c59a0) so that the scorer and the judges here can use them:

- `openai_*`: OpenAI gpt-transcribe, gpt-4o-transcribe, gpt-4o-mini-transcribe and whisper-1, called directly
- `gaik_*`: the same OpenAI models through the gaik toolkit's transcriber
- `vertex_*`: Gemini 3.1 Pro and 3.5 Flash on Vertex AI, prompted to transcribe
- `hh_whisperx_fi`: the Haaga-Helia WhisperX server with Finnish-NLP whisper-large-finnish-v3
- `hf_*`: Whisper large-v3 and Finnish-NLP whisper-large-finnish-v3 with transformers on CSC Roihu

`__r2` and `__r3` are repeat runs of the same arm. The only change from the originals is `input_ref`,
which now points to `data/ajokortti/Ajokortti.mp3`, where `scripts/get_clip.py` puts the clip. The
original run metadata is not copied (it records paths of the private repository). HY's own tool
transcripts, which QAD-1 also scored, are not public and are not here.
