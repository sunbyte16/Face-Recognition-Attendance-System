# 🎯 Face Recognition Attendance System

Detect faces from your webcam, recognize known people using `face_recognition`, and log attendance to a CSV file without duplicate entries per person per day.

<p align="center">
  <a href="#-features">🧩 Features</a> ·
  <a href="#-tech-stack">🛠️ Tech Stack</a> ·
  <a href="#-installation">⚙️ Installation</a> ·
  <a href="#-usage">▶️ Usage</a> ·
  <a href="#-project-structure">🗂️ Structure</a> ·
  <a href="#-roadmap">🗺️ Roadmap</a> ·
  <a href="#-license">📄 License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9--3.11-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/face__recognition-1.3%2B-FF6F00" alt="face_recognition" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-444444" alt="Platforms" />
</p>

---

## 🧩 Features
- ✅ Real‑time face detection and recognition from webcam
- ✅ CSV attendance logging with date, first seen, last seen
- ✅ No duplicate rows per person per day
- ✅ Adjustable tolerance, model (HOG/CNN), and speed/quality trade‑offs
- ✅ Simple folder‑based labeling: add images in `known_faces/`

## 🛠️ Tech Stack
- **OpenCV**: video capture and drawing
- **face_recognition** (dlib): face detection + embeddings + matching
- **NumPy**: vector operations
- **CSV**: lightweight, portable attendance records

## ⚙️ Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
If dlib wheels are needed on Windows, prefer prebuilt wheels to avoid compiling.

## ▶️ Usage
Prepare your labeled images first (see next section), then run:
```bash
python main.py --camera 0 --tolerance 0.5 --attendance-file attendance.csv
```
- **--camera**: webcam index (default: 0)
- **--tolerance**: lower is stricter (0.4–0.6 typical)
- **--upsample**: detection upsample (0–2, higher finds smaller faces)
- **--model**: `hog` (CPU, fast) or `cnn` (GPU, accurate)
- **--resize**: frame downscale for speed (e.g., 0.25)

Press `q` to quit.

## 🗂️ Project Structure
```
.
├─ known_faces/
│  ├─ Alice.jpg
│  └─ Bob/
│     ├─ bob1.jpg
│     └─ bob2.jpg
├─ main.py
├─ requirements.txt
└─ attendance.csv  # created on first run
```
- Filenames or parent folder names are used as labels. Example: `known_faces/Bob/bob1.jpg` → label `Bob`.
- Add multiple, varied images per person (frontal, good lighting) for better accuracy.

## 💡 Tips
- Good lighting and frontal faces improve recognition
- If performance is slow: `--resize 0.25`, `--upsample 0`, and `--model hog`
- Keep `tolerance` around 0.5; lower for fewer false positives, higher for more matches

## 🗺️ Roadmap
- [ ] Optional database backend (SQLite/CSV hybrid)
- [ ] Session summary export (per‑day stats)
- [ ] GUI for managing known faces and parameters

## 📄 License
This project is provided as‑is for learning and personal use. Review applicable licenses of dependencies.

---

## 🙌 Connect
- 🐙 GitHub: [sunbyte16](https://github.com/sunbyte16)
- 🔗 LinkedIn: [Sunil Kumar](https://www.linkedin.com/in/sunil-kumar-bb88bb31a/)
- 🌐 Portfolio: [lively-dodol-cc397c.netlify.app](https://lively-dodol-cc397c.netlify.app)

<p align="center">
  <a href="https://github.com/sunbyte16" target="_blank" rel="noreferrer">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-sunbyte16-181717?logo=github&logoColor=white" />
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/sunil-kumar-bb88bb31a/" target="_blank" rel="noreferrer">
    <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Sunil%20Kumar-0A66C2?logo=linkedin&logoColor=white" />
  </a>
  &nbsp;
  <a href="https://lively-dodol-cc397c.netlify.app" target="_blank" rel="noreferrer">
    <img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-Website-2E7D32?logo=google-chrome&logoColor=white" />
  </a>
</p>

<p align="center">
  Created By <b>❤️Sunil Sharma❤️</b>
</p>

