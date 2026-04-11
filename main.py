from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uuid, os, subprocess, asyncio, math
import edge_tts
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

app = FastAPI()
jobs = {}
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)

whisper_model = None

def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel(
            "base", device="cpu", compute_type="int8"
        )
    return whisper_model

class VideoRequest(BaseModel):
    url: str
    voice: str = "female"
    speed: str = "normal"
    chunk_minutes: int = 30

def update(job_id, msg, progress=None):
    if job_id in jobs:
        jobs[job_id]["message"] = msg
        if progress is not None:
            jobs[job_id]["progress"] = progress
    print(f"[{job_id}] {msg}")

def get_duration(path):
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], capture_output=True, text=True)
        return float(r.stdout.strip())
    except:
        return 0

def download_video(url, out_path):
    # Try different formats
    formats = [
        "best[ext=mp4]",
        "best",
        "bestvideo+bestaudio",
    ]
    for fmt in formats:
        try:
            result = subprocess.run([
                "yt-dlp", "-f", fmt,
                "--merge-output-format", "mp4",
                "-o", out_path, url
            ], capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(out_path):
                return True
        except:
            pass
    raise Exception("Video download failed. Check URL.")

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-y", audio_path
    ], check=True, capture_output=True)

def split_audio(audio_path, job_id, chunk_seconds):
    duration = get_duration(audio_path)
    if duration == 0:
        raise Exception("Could not get audio duration")
    
    total = math.ceil(duration / chunk_seconds)
    chunks = []
    
    for i in range(total):
        start = i * chunk_seconds
        out = f"temp/{job_id}_chunk_{i}.wav"
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-ar", "16000", "-ac", "1",
            "-y", out
        ], capture_output=True)
        
        if os.path.exists(out) and get_duration(out) > 1:
            chunks.append((i, out, start))
    
    return chunks, duration

def transcribe(audio_path):
    try:
        model = get_whisper()
        segments, _ = model.transcribe(
            audio_path, beam_size=5, language="en"
        )
        text = " ".join([s.text for s in segments]).strip()
        return text
    except Exception as e:
        print(f"Transcribe error: {e}")
        return ""

def translate(text):
    try:
        # ছোট ছোট অংশে translate করবে
        words = text.split()
        results = []
        chunk_size = 150
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            try:
                translated = GoogleTranslator(
                    source='auto', target='hi'
                ).translate(chunk)
                results.append(translated)
            except:
                results.append(chunk)
        
        return " ".join(results)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

async def generate_tts(text, out_path, voice, rate):
    words = text.split()
    chunk_size = 300
    temp_files = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        tmp = out_path.replace(".mp3", f"_tmp{i}.mp3")
        try:
            comm = edge_tts.Communicate(chunk, voice, rate=rate)
            await comm.save(tmp)
            if os.path.exists(tmp):
                temp_files.append(tmp)
        except Exception as e:
            print(f"TTS error: {e}")
    
    if not temp_files:
        raise Exception("TTS completely failed")
    
    if len(temp_files) == 1:
        os.rename(temp_files[0], out_path)
    else:
        lst = out_path + ".txt"
        with open(lst, "w") as f:
            for tf in temp_files:
                f.write(f"file '{os.path.abspath(tf)}'\n")
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", lst, "-y", out_path
        ], capture_output=True)
        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)
        if os.path.exists(lst):
            os.remove(lst)

def process_chunk(job_id, idx, chunk_audio, video_path,
                  voice, rate, start, chunk_sec):
    try:
        c = jobs[job_id]["chunks"][idx]
        c["status"] = "processing"

        # 1. Transcribe
        c["step"] = "🎤 Transcribing..."
        text = transcribe(chunk_audio)
        
        if not text or len(text) < 5:
            c["status"] = "done"
            c["step"] = "⚠️ No speech detected"
            return

        # 2. Translate
        c["step"] = "🌐 Translating to Hindi..."
        hindi = translate(text)
        c["hindi_text"] = hindi

        # 3. TTS
        c["step"] = "🔊 Generating Hindi voice..."
        dubbed = f"temp/{job_id}_{idx}_dub.mp3"
        asyncio.run(generate_tts(hindi, dubbed, voice, rate))

        # 4. Cut video
        c["step"] = "✂️ Cutting video..."
        chunk_dur = get_duration(chunk_audio)
        vid_seg = f"temp/{job_id}_{idx}_vid.mp4"
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start),
            "-t", str(chunk_dur),
            "-c:v", "copy", "-an",
            "-y", vid_seg
        ], capture_output=True)

        # 5. Merge
        c["step"] = "🔗 Merging..."
        out = f"static/{job_id}_part{idx+1}.mp4"
        subprocess.run([
            "ffmpeg",
            "-i", vid_seg,
            "-i", dubbed,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest", "-y", out
        ], capture_output=True)

        if os.path.exists(out) and os.path.getsize(out) > 0:
            c["status"] = "done"
            c["step"] = "✅ Ready to download!"
            c["download_url"] = f"/download/{job_id}/{idx}"
        else:
            raise Exception("Output file not created")

        # Cleanup
        for f in [chunk_audio, dubbed, vid_seg]:
            if os.path.exists(f):
                os.remove(f)

    except Exception as e:
        jobs[job_id]["chunks"][idx]["status"] = "failed"
        jobs[job_id]["chunks"][idx]["step"] = f"❌ {str(e)}"

def run_job(job_id, url, voice, speed, chunk_minutes):
    try:
        jobs[job_id]["status"] = "running"
        update(job_id, "📥 Downloading video...", 5)

        video = f"temp/{job_id}.mp4"
        audio = f"temp/{job_id}.wav"

        download_video(url, video)
        
        update(job_id, "🔊 Extracting audio...", 15)
        extract_audio(video, audio)

        update(job_id, "✂️ Splitting into parts...", 20)
        chunk_sec = chunk_minutes * 60
        chunks, duration = split_audio(audio, job_id, chunk_sec)
        total = len(chunks)

        if total == 0:
            raise Exception("No audio chunks created")

        jobs[job_id]["total_chunks"] = total
        jobs[job_id]["chunk_minutes"] = chunk_minutes
        jobs[job_id]["chunks"] = [
            {
                "index": i,
                "status": "waiting",
                "step": "⏳ Waiting...",
                "download_url": None,
                "hindi_text": ""
            }
            for i in range(total)
        ]

        v = "hi-IN-MadhurNeural" if voice == "male" else "hi-IN-SwaraNeural"
        r = "-20%" if speed == "slow" else "+20%" if speed == "fast" else "+0%"

        # একটার পর একটা process
        for idx, chunk_path, start in chunks:
            prog = 25 + int((idx / total) * 70)
            update(job_id, f"⚙️ Processing part {idx+1} of {total}...", prog)
            process_chunk(job_id, idx, chunk_path, video, v, r, start, chunk_sec)

        if os.path.exists(audio):
            os.remove(audio)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "🎉 All parts done!"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"❌ {str(e)}"

@app.post("/start")
def start(req: VideoRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "starting", "progress": 0,
        "message": "Starting...", "chunks": [],
        "total_chunks": 0, "chunk_minutes": req.chunk_minutes
    }
    bg.add_task(run_job, job_id, req.url,
                req.voice, req.speed, req.chunk_minutes)
    return {"job_id": job_id}

@app.post("/retry/{job_id}/{idx}")
def retry(job_id: str, idx: int, bg: BackgroundTasks):
    if job_id not in jobs:
        return {"error": "Not found"}
    job = jobs[job_id]
    cm = job.get("chunk_minutes", 30)
    cs = cm * 60
    start = idx * cs
    video = f"temp/{job_id}.mp4"
    audio = f"temp/{job_id}.wav"
    chunk_audio = f"temp/{job_id}_chunk_{idx}.wav"

    subprocess.run([
        "ffmpeg", "-i", audio,
        "-ss", str(start), "-t", str(cs),
        "-ar", "16000", "-ac", "1",
        "-y", chunk_audio
    ], capture_output=True)

    job["chunks"][idx]["status"] = "processing"
    job["chunks"][idx]["step"] = "🔄 Retrying..."
    bg.add_task(process_chunk, job_id, idx, chunk_audio,
                video, "hi-IN-SwaraNeural", "+0%", start, cs)
    return {"ok": True}

@app.get("/status/{job_id}")
def status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

@app.get("/download/{job_id}/{idx}")
def download(job_id: str, idx: int):
    path = f"static/{job_id}_part{idx+1}.mp4"
    if os.path.exists(path):
        return FileResponse(path, filename=f"hindi_part{idx+1}.mp4")
    return {"error": "Not found"}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")