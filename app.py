"""
Noise Detector - Monitor ambient noise and detect human speech
Uses Silero VAD for speech detection, Whisper for transcription, logs to SQLite
"""

import threading
import queue
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import mlx_whisper
from scipy.io import wavfile

from fasthtml.common import *

# --- Database Setup ---
db = database("noise_events.db")

@dataclass
class NoiseEvent:
    id: int = None
    timestamp: str = ""
    db_level: float = 0.0
    is_speech: bool = False
    duration_ms: int = 0
    transcription: str = ""

events = db.create(NoiseEvent, pk="id", transform=True)

# --- Audio Config ---
SAMPLE_RATE = 16000  # Required for both Silero VAD and Whisper
CHUNK_DURATION = 2.0  # Longer chunks for better transcription (2 seconds)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
DB_THRESHOLD = 15  # Only log events above this dB level

# --- Silero VAD Setup ---
torch.set_num_threads(1)
vad_model = load_silero_vad()

# --- Whisper Model (lazy load) ---
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
whisper_loaded = {"ready": False}

# --- Transcription Queue ---
transcription_queue = queue.Queue()

# --- Monitoring State ---
monitoring = {"active": False, "thread": None, "transcribe_thread": None}

def calculate_db(audio_chunk):
    """Calculate dB SPL from audio samples"""
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    if rms > 0:
        db = 20 * np.log10(rms) + 80
        return max(0, min(120, db))
    return 0

def detect_speech(audio_chunk):
    """Use Silero VAD to detect speech"""
    try:
        audio_tensor = torch.from_numpy(audio_chunk).float()
        timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=SAMPLE_RATE)
        return len(timestamps) > 0
    except Exception as e:
        print(f"VAD error: {e}")
        return False

def transcribe_audio(audio_chunk):
    """Transcribe audio using Whisper V3 Turbo"""
    try:
        # Save to temp wav file (mlx-whisper prefers file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Ensure audio is in correct format
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            wavfile.write(f.name, SAMPLE_RATE, audio_int16)

            result = mlx_whisper.transcribe(
                f.name,
                path_or_hf_repo=WHISPER_MODEL,
                language="en",
            )

            text = result.get("text", "").strip()
            return text if text else ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def transcription_worker():
    """Background worker that processes transcription queue"""
    print("Transcription worker started...")
    while monitoring["active"]:
        try:
            # Wait for items with timeout so we can check if still active
            item = transcription_queue.get(timeout=1.0)
            event_id, audio_chunk = item

            print(f"Transcribing event {event_id}...")
            text = transcribe_audio(audio_chunk)

            if text:
                # Update the event with transcription
                event = events[event_id]
                event.transcription = text
                events.update(event)
                print(f"Transcription: {text[:50]}...")

            transcription_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Transcription worker error: {e}")

    print("Transcription worker stopped")

def audio_monitor_loop():
    """Background thread that captures and analyzes audio"""
    print("Starting audio monitor...")

    def audio_callback(indata, frames, time_info, status):
        if not monitoring["active"]:
            return

        audio = indata[:, 0].copy()
        db_level = calculate_db(audio)

        if db_level >= DB_THRESHOLD:
            is_speech = detect_speech(audio)

            # Insert event first
            event = events.insert(NoiseEvent(
                timestamp=datetime.now().isoformat(),
                db_level=float(round(db_level, 1)),
                is_speech=bool(is_speech),
                duration_ms=int(CHUNK_DURATION * 1000),
                transcription=""
            ))

            # Queue for transcription (regardless of VAD result, as user requested)
            transcription_queue.put((event.id, audio.copy()))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        while monitoring["active"]:
            time.sleep(0.1)

    print("Audio monitor stopped")

# --- FastHTML App ---
app, rt = fast_app(
    hdrs=[
        Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
        Style("""
            body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
            .stat-card { background: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; }
            .stat-label { color: #666; margin-top: 5px; }
            .controls { margin: 20px 0; }
            .btn { padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; border-radius: 5px; }
            .btn-start { background: #22c55e; color: white; }
            .btn-stop { background: #ef4444; color: white; }
            .event-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            .event-table th, .event-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            .speech { color: #ef4444; font-weight: bold; }
            .noise { color: #666; }
            .transcription { font-style: italic; color: #333; max-width: 400px; }
            #chart-container { height: 300px; margin: 20px 0; }
            .status-active { color: #22c55e; }
            .status-inactive { color: #666; }
            .queue-info { color: #666; font-size: 0.9em; }
        """)
    ]
)

@rt
def index():
    return Titled(
        "Noise Detector",
        Div(id="dashboard", hx_get="/dashboard", hx_trigger="load, every 2s"),
    )

@rt
def dashboard():
    recent = list(events(order_by="-id", limit=100))
    total = len(list(events()))
    speech_events = len([e for e in recent if e.is_speech])
    noise_events = len([e for e in recent if not e.is_speech])
    avg_db = sum(float(e.db_level) for e in recent) / len(recent) if recent else 0
    queue_size = transcription_queue.qsize()

    status_class = "status-active" if monitoring["active"] else "status-inactive"
    status_text = "MONITORING" if monitoring["active"] else "STOPPED"

    return Div(
        Div(
            H2(f"Status: ", Span(status_text, cls=status_class)),
            P(f"Transcription queue: {queue_size} pending", cls="queue-info") if queue_size > 0 else None,
            Div(
                Button("Start Monitoring", cls="btn btn-start", hx_post="/start", hx_target="#dashboard") if not monitoring["active"] else
                Button("Stop Monitoring", cls="btn btn-stop", hx_post="/stop", hx_target="#dashboard"),
                Button("Clear Logs", cls="btn", style="margin-left: 10px; background: #666; color: white;", hx_post="/clear", hx_target="#dashboard"),
                cls="controls"
            )
        ),
        Div(
            Div(Div(f"{total}", cls="stat-value"), Div("Total Events", cls="stat-label"), cls="stat-card"),
            Div(Div(f"{speech_events}", cls="stat-value speech"), Div("Speech Events", cls="stat-label"), cls="stat-card"),
            Div(Div(f"{noise_events}", cls="stat-value"), Div("Noise Events", cls="stat-label"), cls="stat-card"),
            Div(Div(f"{avg_db:.1f} dB", cls="stat-value"), Div("Avg Level", cls="stat-label"), cls="stat-card"),
            cls="stats"
        ),
        Div(
            Canvas(id="noiseChart"),
            Script(render_chart_js(recent)),
            id="chart-container"
        ),
        H3("Recent Events"),
        Table(
            Thead(Tr(Th("Time"), Th("Level"), Th("Type"), Th("Transcription"))),
            Tbody(*[
                Tr(
                    Td(e.timestamp.split("T")[1][:8] if "T" in e.timestamp else e.timestamp),
                    Td(f"{e.db_level} dB"),
                    Td("SPEECH", cls="speech") if e.is_speech else Td("noise", cls="noise"),
                    Td(e.transcription[:100] + "..." if len(e.transcription) > 100 else e.transcription, cls="transcription") if e.transcription else Td("-", style="color: #ccc;")
                ) for e in recent[:20]
            ]),
            cls="event-table"
        )
    )

def render_chart_js(events_list):
    events_list = list(reversed(events_list[:50]))
    labels = [e.timestamp.split("T")[1][:8] if "T" in e.timestamp else "" for e in events_list]
    db_values = [float(e.db_level) for e in events_list]
    colors = ["rgba(239, 68, 68, 0.8)" if e.is_speech else "rgba(100, 100, 100, 0.5)" for e in events_list]

    return f"""
    if (window.noiseChart) {{ window.noiseChart.destroy(); }}
    window.noiseChart = new Chart(document.getElementById('noiseChart'), {{
        type: 'bar',
        data: {{
            labels: {labels},
            datasets: [{{
                label: 'dB Level',
                data: {db_values},
                backgroundColor: {colors},
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'dB' }} }}
            }},
            plugins: {{ legend: {{ display: false }} }}
        }}
    }});
    """

@rt
def start():
    if not monitoring["active"]:
        monitoring["active"] = True
        # Start audio monitor thread
        monitoring["thread"] = threading.Thread(target=audio_monitor_loop, daemon=True)
        monitoring["thread"].start()
        # Start transcription worker thread
        monitoring["transcribe_thread"] = threading.Thread(target=transcription_worker, daemon=True)
        monitoring["transcribe_thread"].start()
    return RedirectResponse("/dashboard", status_code=303)

@rt
def stop():
    monitoring["active"] = False
    return RedirectResponse("/dashboard", status_code=303)

@rt
def clear():
    for e in events():
        events.delete(e.id)
    return RedirectResponse("/dashboard", status_code=303)

serve()
