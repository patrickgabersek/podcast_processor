import os
import shutil
import tempfile

import anthropic
import streamlit as st
import torch
import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# ========================= CONFIG =========================
st.set_page_config(page_title="YouTube Podcast → Transcript + Claude Summary", layout="wide")
st.title("🎙️ YouTube Podcast Processor")
st.markdown("Paste a YouTube URL → get transcript + Claude summary")

# Claude client — picks up ANTHROPIC_API_KEY from env automatically
client = anthropic.Anthropic()

# Max characters of transcript to send to Claude (~100k tokens safety cap)
MAX_TRANSCRIPT_CHARS = 80_000


# ====================== DOWNLOAD AUDIO ======================
def download_audio(youtube_url: str) -> tuple[str, str, str]:
    """Download best audio from YouTube. Returns (audio_path, title, temp_dir)."""
    temp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{temp_dir}/%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "0",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get("title", "podcast")
            audio_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"

        return audio_path, title, temp_dir

    except Exception:
        # Clean up immediately if download fails so the temp dir doesn't leak
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# ====================== WHISPER MODEL ======================
@st.cache_resource
def load_whisper_model() -> WhisperModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    label = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Loading Whisper model on: {device.upper()} ({label})")
    return WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)


# ====================== TRANSCRIBE ======================
def transcribe_audio(audio_path: str, progress_bar) -> tuple[str, str, str]:
    """
    Transcribe audio file. Streams segments so we can show progress.
    Returns (timed_transcript, plain_text, language).
    """
    model = load_whisper_model()
    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=False,
    )

    timed_lines = []
    plain_parts = []
    duration = info.duration or 1  # avoid division by zero

    for seg in segments_gen:
        timed_lines.append(f"[{seg.start:.0f}s - {seg.end:.0f}s] {seg.text.strip()}")
        plain_parts.append(seg.text.strip())
        progress_bar.progress(min(seg.end / duration, 1.0))

    return "\n".join(timed_lines), " ".join(plain_parts), info.language


# ====================== CLAUDE SUMMARY ======================
def get_claude_summary_streamed(transcript_text: str, title: str, placeholder):
    """
    Stream Claude summary into a Streamlit placeholder.
    Truncates transcript if it exceeds MAX_TRANSCRIPT_CHARS to stay within context limits.
    """
    truncated = False
    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        transcript_text = transcript_text[:MAX_TRANSCRIPT_CHARS]
        truncated = True

    prompt = f"""Summarize this podcast transcript in clean Markdown.

Title: {title}
{"⚠️ Note: transcript was truncated to fit context limits." if truncated else ""}

Transcript:
{transcript_text}

Return exactly these sections:
- **One-sentence hook**
- **Executive Summary** (2-3 sentences max)
- **Key Takeaways** (6-10 bullets)
- **Notable Quotes** (3-4 best quotes)
- **Actionable Insights** (3-5 bullets)

Be concise, direct, and accurate. No fluff."""

    collected = []
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system="You are a concise podcast summarizer.",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text_chunk in stream.text_stream:
            collected.append(text_chunk)
            placeholder.markdown("".join(collected))

    return "".join(collected)


# ====================== STREAMLIT UI ======================
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Process Podcast", type="primary"):
    if not url.strip():
        st.error("Please paste a URL")
        st.stop()

    # --- Download ---
    with st.status("Downloading audio...", expanded=False) as status:
        try:
            audio_path, video_title, temp_dir = download_audio(url)
            status.update(label=f"✅ Downloaded: {video_title}", state="complete")
        except Exception as e:
            status.update(label="❌ Download failed", state="error")
            st.error(f"Download error: {e}")
            st.stop()

    # --- Transcribe + Summarize (always clean up temp files) ---
    try:
        with st.status("Transcribing with faster-whisper...", expanded=True):
            prog = st.progress(0.0, text="Transcribing segments...")
            timed_transcript, plain_text, lang = transcribe_audio(audio_path, prog)
            prog.progress(1.0, text="Transcription complete")

        st.success(f"✅ Transcription done — language: **{lang}**")

        if len(plain_text) > MAX_TRANSCRIPT_CHARS:
            st.warning(
                f"⚠️ Transcript is very long ({len(plain_text):,} chars). "
                f"Sending first {MAX_TRANSCRIPT_CHARS:,} chars to Claude."
            )

        with st.status("Summarizing with Claude...", expanded=True):
            summary_placeholder = st.empty()
            summary = get_claude_summary_streamed(plain_text, video_title, summary_placeholder)

    finally:
        # Always clean up, even if transcription/summarization throws
        if os.path.exists(audio_path):
            os.remove(audio_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

    # --- Results ---
    st.divider()
    tab1, tab2 = st.tabs(["📜 Full Transcript", "📋 Claude Summary"])

    with tab1:
        st.text_area("Timed Transcript", timed_transcript, height=600)
        st.download_button(
            "⬇️ Download transcript.txt",
            timed_transcript,
            file_name=f"{video_title}_transcript.txt",
        )

    with tab2:
        st.markdown(summary)
        st.download_button(
            "⬇️ Download summary.md",
            summary,
            file_name=f"{video_title}_summary.md",
        )

    st.balloons()

st.caption("Built with yt-dlp + faster-whisper + Claude Sonnet 4 • Runs locally")
