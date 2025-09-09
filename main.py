import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import face_recognition
import numpy as np


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
	parser.add_argument("--known-faces", default="known_faces", help="Directory of labeled images")
	parser.add_argument("--camera", type=int, default=0, help="Webcam index")
	parser.add_argument("--tolerance", type=float, default=0.5, help="Face match tolerance (lower is stricter)")
	parser.add_argument("--upsample", type=int, default=1, help="Detection upsample factor (0-2)")
	parser.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Face detection model")
	parser.add_argument("--attendance-file", default="attendance.csv", help="CSV file to write attendance")
	parser.add_argument("--resize", type=float, default=0.25, help="Frame downscale factor for speed (e.g., 0.25)")
	return parser.parse_args()


def list_image_files(directory: Path) -> List[Path]:
	image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
	files: List[Path] = []
	for root, _, filenames in os.walk(directory):
		for filename in filenames:
			if Path(filename).suffix.lower() in image_extensions:
				files.append(Path(root) / filename)
	return files


def extract_label_from_path(image_path: Path, base_dir: Path) -> str:
	try:
		rel = image_path.relative_to(base_dir)
	except Exception:
		rel = image_path.name
	parts = Path(rel).parts
	if len(parts) >= 2:  # subfolder structure
		label = parts[-2]
	else:
		label = Path(rel).stem
	return label


def load_known_faces(known_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
	encodings: List[np.ndarray] = []
	labels: List[str] = []
	if not known_dir.exists():
		print(f"[INFO] Known faces directory not found: {known_dir}. Creating it.")
		known_dir.mkdir(parents=True, exist_ok=True)
		return encodings, labels

	image_paths = list_image_files(known_dir)
	if not image_paths:
		print(f"[WARN] No images found in {known_dir}. Add images and restart.")
		return encodings, labels

	for img_path in image_paths:
		label = extract_label_from_path(img_path, known_dir)
		image = face_recognition.load_image_file(str(img_path))
		face_locations = face_recognition.face_locations(image, model="hog")
		if not face_locations:
			print(f"[WARN] No face found in image: {img_path}")
			continue
		face_encs = face_recognition.face_encodings(image, face_locations)
		if not face_encs:
			print(f"[WARN] Could not compute encoding: {img_path}")
			continue
		encodings.append(face_encs[0])
		labels.append(label)
		print(f"[LOAD] {label}: {img_path}")
	return encodings, labels


def ensure_attendance_header(file_path: Path) -> None:
	if not file_path.exists():
		file_path.parent.mkdir(parents=True, exist_ok=True)
		with file_path.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["date", "name", "first_seen", "last_seen"])  # ISO times


def read_today_attendance(file_path: Path) -> Dict[str, Dict[str, str]]:
	attendance: Dict[str, Dict[str, str]] = {}
	if not file_path.exists():
		return attendance
		today = datetime.now().date().isoformat()
	with file_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if row.get("date") == datetime.now().date().isoformat():
				name = row.get("name", "")
				attendance[name] = {
					"date": row.get("date", ""),
					"first_seen": row.get("first_seen", ""),
					"last_seen": row.get("last_seen", ""),
				}
	return attendance


def write_attendance_snapshot(file_path: Path, attendance_today: Dict[str, Dict[str, str]]) -> None:
	# Rewrite all rows except today's for others, then append today's snapshot rows.
	rows: List[Dict[str, str]] = []
	if file_path.exists():
		with file_path.open("r", newline="", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for row in reader:
				if row.get("date") != datetime.now().date().isoformat():
					rows.append(row)
	# add today's rows from snapshot
	for name, data in attendance_today.items():
		rows.append({
			"date": data["date"],
			"name": name,
			"first_seen": data["first_seen"],
			"last_seen": data["last_seen"],
		})
	with file_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["date", "name", "first_seen", "last_seen"])
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	args = parse_arguments()

	known_dir = Path(args.known_faces)
	attendance_file = Path(args.attendance_file)

	print("[INFO] Loading known faces...")
	known_encodings, known_labels = load_known_faces(known_dir)
	print(f"[INFO] Loaded {len(known_encodings)} known faces")

	ensure_attendance_header(attendance_file)
	attendance_today = read_today_attendance(attendance_file)

	video = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
	if not video.isOpened():
		raise RuntimeError(f"Cannot open camera index {args.camera}")

	frame_resize = float(args.resize)
	if frame_resize <= 0 or frame_resize > 1:
		frame_resize = 1.0

	print("[INFO] Starting video. Press 'q' to quit.")
	while True:
		ret, frame = video.read()
		if not ret:
			print("[WARN] Failed to read frame")
			break

		# Resize for speed
		small_frame = cv2.resize(frame, (0, 0), fx=frame_resize, fy=frame_resize)
		rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

		# Detect faces
		face_locations = face_recognition.face_locations(
			rgb_small_frame, number_of_times_to_upsample=args.upsample, model=args.model
		)
		face_encodings = []
		if face_locations:
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names: List[str] = []
		for face_encoding in face_encodings:
			name = "Unknown"
			if known_encodings:
				matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=args.tolerance)
				distances = face_recognition.face_distance(known_encodings, face_encoding)
				best_match_index = int(np.argmin(distances)) if len(distances) else -1
				if best_match_index >= 0 and matches[best_match_index]:
					name = known_labels[best_match_index]
			face_names.append(name)

		# Attendance update
		now = datetime.now()
		today = now.date().isoformat()
		for name in face_names:
			if name == "Unknown":
				continue
			first_seen = attendance_today.get(name, {}).get("first_seen")
			if not first_seen:
				attendance_today[name] = {
					"date": today,
					"first_seen": now.isoformat(timespec="seconds"),
					"last_seen": now.isoformat(timespec="seconds"),
				}
			else:
				attendance_today[name]["last_seen"] = now.isoformat(timespec="seconds")

		# Draw results
		for (top, right, bottom, left), name in zip(face_locations, face_names):
			# scale back up if resized
			scale = 1.0 / frame_resize
			top = int(top * scale)
			right = int(right * scale)
			bottom = int(bottom * scale)
			left = int(left * scale)
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
			cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
			cv2.putText(frame, name, (left + 4, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		cv2.imshow("Attendance", frame)

		# Periodically write snapshot (every frame is fine for simplicity)
		write_attendance_snapshot(attendance_file, attendance_today)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	video.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
