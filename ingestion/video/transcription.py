"""
transcription.py
=================

Vimeo transcription (كان transcribe_fetch.py). الفرق الوحيد عن النسخة
القديمة: الدوال بترجّع الـ dict في الميموري (مش بس بتكتبه ملف)، عشان
video_pipeline.py يقدر يمرره على طول لـ video_chunking.py من غير ما
يعدّي على raw_extracted/*.json كملف وسيط - لكن fetch_and_save() لسه
موجودة لو حابة تحتفظي بنسخة خام على الدِسك للتصحيح/الأرشفة.
"""
import os
import re
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
VIMEO_TOKEN = os.getenv("VIMEO_TOKEN")
PREFERRED_LANGUAGE = "ar"          # لو مش لاقي اللغة دي هياخد أول ترانسكريبت متاح
OUTPUT_DIR = Path("raw_extracted")  # لأرشفة اختيارية بس - مش خطوة إلزامية في الـ pipeline

# لو True: أي فيديو معندوش ترانسكريبت، هيطلب من Vimeo AI يعمله تلقائي وينتظره
# (محتاج ai scope + خطة Enterprise)
GENERATE_IF_MISSING = False
AI_TRANSCRIBE_LANGUAGE = None  # None = auto-detect اللغة من صوت الفيديو

API_BASE = "https://api.vimeo.com"


def _headers():
    return {"Authorization": f"Bearer {VIMEO_TOKEN}"}


# ============ 1) جلب المجلدات والفيديوهات ============

def list_folders() -> list[dict]:
    """بيرجع [{id, name}, ...] لكل المجلدات (Folders) في حسابك على Vimeo."""
    folders = []
    url = f"{API_BASE}/me/folders"
    params = {"per_page": 100}

    while url:
        resp = requests.get(url, headers=_headers(), params=params)
        resp.raise_for_status()
        data = resp.json()
        for f in data.get("data", []):
            folder_id = f["uri"].split("/")[-1]
            folders.append({"id": folder_id, "name": f.get("name", folder_id)})
        url = data.get("paging", {}).get("next")
        params = {}

    return folders


def get_videos_from_folder(folder_id: str) -> list[dict]:
    """بيرجع [{id, name}, ...] لكل الفيديوهات جوه مجلد (Folder/Project) معين."""
    videos = []
    url = f"{API_BASE}/me/projects/{folder_id}/videos"
    params = {"per_page": 100}

    while url:
        resp = requests.get(url, headers=_headers(), params=params)
        resp.raise_for_status()
        data = resp.json()
        for v in data.get("data", []):
            video_id = v["uri"].split("/")[-1]
            videos.append({"id": video_id, "name": v.get("name") or video_id})
        url = data.get("paging", {}).get("next")
        params = {}

    return videos


# ============ 2) توليد ترانسكريبت جديد بالـ AI (لو الفيديو معندوش واحد) ============

def request_ai_transcription(video_id: str, language: str | None = None) -> dict:
    """
    بيطلب من Vimeo AI يعمل ترانسكريبت جديد للفيديو من الصوت. محتاج توكن
    بصلاحية "ai" (خطة Enterprise). لو الفيديو عنده ترانسكريبت بالفعل،
    بيرجع {"already_exists": True} من غير error.
    """
    url = f"{API_BASE}/videos/{video_id}/ai/transcribe"
    body = {"language": language} if language else {}

    resp = requests.post(url, headers=_headers(), json=body)

    if resp.status_code == 400:
        error_code = resp.json().get("error_code")
        if error_code == 3990:
            return {"already_exists": True}
        if error_code == 3991:
            return {"in_progress": True}

    resp.raise_for_status()
    return resp.json()


def poll_ai_transcription(video_id: str, interval: int = 5, timeout: int = 300) -> dict:
    """بيستنى لحد ما Vimeo AI يخلص توليد الترانسكريبت."""
    url = f"{API_BASE}/videos/{video_id}/ai/transcribe"
    waited = 0

    while waited < timeout:
        resp = requests.get(url, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        if status in ("complete", "finished", "ready"):
            return data
        if status == "error":
            raise RuntimeError(f"AI transcription failed for video {video_id}: {data}")

        time.sleep(interval)
        waited += interval

    raise TimeoutError(f"AI transcription timed out for video {video_id} after {timeout}s")


def ensure_ai_transcript(video_id: str, language: str | None = AI_TRANSCRIBE_LANGUAGE) -> None:
    """بيتأكد إن الفيديو عنده ترانسكريبت: لو معندوش، بيطلب من Vimeo AI يعمله وينتظره."""
    result = request_ai_transcription(video_id, language)

    if result.get("already_exists"):
        return
    if result.get("in_progress"):
        print(f"  - Transcription already in progress for {video_id}, waiting...")
        poll_ai_transcription(video_id)
        return

    print(f"  - Requested AI transcription for {video_id}, waiting...")
    poll_ai_transcription(video_id)
    print(f"  - AI transcription complete for {video_id}")


# ============ 3) جلب رابط الترانسكريبت لفيديو واحد ============

def get_texttrack(video_id: str, preferred_lang: str = PREFERRED_LANGUAGE) -> dict | None:
    url = f"{API_BASE}/videos/{video_id}/texttracks"
    resp = requests.get(url, headers=_headers())
    resp.raise_for_status()
    tracks = resp.json().get("data", [])

    if not tracks:
        return None

    for t in tracks:
        if t.get("language") == preferred_lang:
            return t

    return tracks[0]  # مفيش نسخة باللغة المفضلة، ناخد أول واحدة متاحة


# ============ 4) تحميل ملف VTT وتحويله لـ segments ============

def download_vtt(link: str) -> str:
    resp = requests.get(link)
    resp.raise_for_status()
    return resp.text


_TIME_PATTERN = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def parse_vtt(vtt_text: str) -> list[dict]:
    """بيحوّل محتوى ملف VTT لقائمة [{timestamp_s, end_s, text}, ...]."""
    lines = vtt_text.splitlines()
    segments = []
    i = 0

    while i < len(lines):
        match = _TIME_PATTERN.search(lines[i])
        if match:
            h1, m1, s1, ms1 = (int(match.group(1)), int(match.group(2)),
                               int(match.group(3)), int(match.group(4)))
            h2, m2, s2, ms2 = (int(match.group(5)), int(match.group(6)),
                               int(match.group(7)), int(match.group(8)))
            start_s = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            end_s = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            i += 1

            text_lines = []
            while i < len(lines) and lines[i].strip() != "":
                text_lines.append(lines[i].strip())
                i += 1

            text = " ".join(text_lines).strip()
            if text:
                segments.append({
                    "timestamp_s": round(start_s, 1),
                    "end_s": round(end_s, 1),
                    "text": text,
                })
        else:
            i += 1

    return segments


# ============ 5) الجلب - النسخة الأساسية اللي video_pipeline.py بيستخدمها ============

def fetch_transcript(video_id: str, video_name: str | None = None) -> dict | None:
    """
    بيرجع dict {source_name, source_type, pages} جاهز يتبعت على طول لـ
    video_chunking.chunk_video_transcript() - من غير كتابة أي ملف.
    بيرجع None لو الفيديو معندوش ترانسكريبت.
    """
    if GENERATE_IF_MISSING:
        ensure_ai_transcript(video_id)

    track = get_texttrack(video_id)
    if not track:
        print(f"  ! No transcript found for video {video_id}")
        return None

    vtt_text = download_vtt(track["link"])
    segments = parse_vtt(vtt_text)

    return {
        "source_name": video_name or video_id,
        "source_type": "video",
        "pages": segments,
        "language": track.get("language"),
    }


def fetch_and_save(video_id: str, video_name: str | None = None) -> dict | None:
    """
    نفس fetch_transcript() لكن كمان بتكتب نسخة خام في raw_extracted/ -
    مفيدة لو عايزة أرشيف/تصحيح مستقل عن باقي الـ pipeline.
    """
    data = fetch_transcript(video_id, video_name)
    if data is None:
        return None

    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^\w\-. ]", "_", data["source_name"])
    out_path = OUTPUT_DIR / f"{safe_name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  - Saved: {out_path.name} ({len(data['pages'])} segments, language={data.get('language')})")
    return data


if __name__ == "__main__":
    # سحب كل فيديوهات مجلد معين أوتوماتيك وحفظهم كنسخة خام (أرشفة/تصحيح)
    FOLDER_ID = "30332812"  # Scholaria_videos

    videos = get_videos_from_folder(FOLDER_ID)
    print(f"Found {len(videos)} videos in folder {FOLDER_ID}")

    for v in videos:
        print(f"Fetching transcript for {v['name']} ...")
        try:
            fetch_and_save(v["id"], v["name"])
        except Exception as e:
            print(f"  ! Error for {v['name']}: {e}")

    print("\nDone, all JSON files saved in raw_extracted/")
