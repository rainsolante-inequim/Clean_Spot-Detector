import os
import shutil
import threading
import webbrowser
import hashlib
import numpy as np
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
from predict import predict_waste

app = Flask(__name__)
app.secret_key = "cleanspot-development-key"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "cleanspot"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mysql = MySQL(app)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}
MAX_REMOTE_IMAGE_BYTES = 8 * 1024 * 1024

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def download_remote_image(image_url):
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Only HTTP and HTTPS image URLs are allowed.")

    request = Request(image_url, headers={"User-Agent": "CleanSpot dataset importer"})
    with urlopen(request, timeout=15) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError("URL did not return an image.")
        image_bytes = response.read(MAX_REMOTE_IMAGE_BYTES + 1)

    if len(image_bytes) > MAX_REMOTE_IMAGE_BYTES:
        raise ValueError("Image is larger than 8 MB.")
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the downloaded image.")
    return image_bytes, image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/checker", methods=["GET", "POST"])
def checker():
    results = None
    if request.method == "POST":
        item = request.form.get("item", "").strip()
        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT item_name, category, disposal_method FROM waste_items "
            "WHERE item_name LIKE %s ORDER BY item_name",
            (f"%{item}%",)
        )
        results = cursor.fetchall()
        cursor.close()
    return render_template("checker.html", results=results)

@app.route("/guide")
def guide():
    cursor = mysql.connection.cursor()
    cursor.execute(
        "SELECT item_name, category, disposal_method FROM waste_items "
        "ORDER BY category, item_name"
    )
    wastes = cursor.fetchall()
    cursor.close()
    return render_template("guide.html", wastes=wastes)

@app.route("/scan")
def scan():
    return render_template("scan.html")

@app.route("/dataset/add", methods=["POST"])
def add_dataset_image():
    image = request.files.get("image")
    category = request.form.get("category", "").strip().lower()
    categories = {"biodegradable", "recyclable", "residual", "special"}

    if category not in categories:
        flash("Choose a valid waste category before saving the image.", "warning")
        return redirect(url_for("scan"))

    if not image or not image.filename or not allowed_file(image.filename):
        flash("Dataset images must be JPG, JPEG, or PNG files.", "warning")
        return redirect(url_for("scan"))

    category_folder = os.path.join(DATASET_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)
    extension = image.filename.rsplit(".", 1)[1].lower()
    filename = f"webcam_{uuid4().hex}.{extension}"
    image.save(os.path.join(category_folder, filename))
    flash(f"Captured image added to the {category} dataset. Retrain the model when ready.", "success")
    return redirect(url_for("scan"))

@app.route("/dataset/video", methods=["POST"])
def add_dataset_video():
    video = request.files.get("video")
    category = request.form.get("video_category", "").strip().lower()
    categories = {"biodegradable", "recyclable", "residual", "special"}

    if category not in categories:
        flash("Choose a valid waste category before importing video frames.", "warning")
        return redirect(url_for("scan"))
    if not video or not video.filename or not allowed_video(video.filename):
        flash("Use an MP4, MOV, AVI, WEBM, or MKV video.", "warning")
        return redirect(url_for("scan"))

    temporary_path = os.path.join(app.config["UPLOAD_FOLDER"], f"video_{uuid4().hex}.tmp")
    video.save(temporary_path)
    category_folder = os.path.join(DATASET_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)
    capture = cv2.VideoCapture(temporary_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = max(1, frame_count // 100)
    saved = 0
    frame_number = 0

    try:
        while saved < 100:
            success, frame = capture.read()
            if not success:
                break
            if frame_number % sample_step == 0:
                filename = f"video_{uuid4().hex}.jpg"
                if cv2.imwrite(os.path.join(category_folder, filename), frame):
                    saved += 1
            frame_number += 1
    finally:
        capture.release()
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    if saved == 0:
        flash("No readable frames were found in that video.", "warning")
    else:
        flash(f"Added {saved} video frames to the {category} dataset. Review them before training.", "success")
    return redirect(url_for("scan"))

@app.route("/dataset/images", methods=["POST"])
def add_dataset_images():
    images = request.files.getlist("images")
    category = request.form.get("images_category", "").strip().lower()
    categories = {"biodegradable", "recyclable", "residual", "special"}

    if category not in categories:
        flash("Choose a valid waste category before importing pictures.", "warning")
        return redirect(url_for("scan"))

    category_folder = os.path.join(DATASET_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)
    saved = 0
    for image in images:
        if not image or not image.filename or not allowed_file(image.filename):
            continue
        extension = image.filename.rsplit(".", 1)[1].lower()
        filename = f"internet_{uuid4().hex}.{extension}"
        image.save(os.path.join(category_folder, filename))
        saved += 1

    if saved:
        flash(f"Added {saved} pictures to the {category} dataset. Review them before training.", "success")
    else:
        flash("No valid JPG, JPEG, or PNG pictures were selected.", "warning")
    return redirect(url_for("scan"))

@app.route("/dataset/urls", methods=["POST"])
def add_dataset_urls():
    category = request.form.get("url_category", "").strip().lower()
    image_urls = [url.strip() for url in request.form.get("image_urls", "").splitlines() if url.strip()]
    categories = {"biodegradable", "recyclable", "residual", "special"}

    if category not in categories:
        flash("Choose a valid waste category before importing URLs.", "warning")
        return redirect(url_for("scan"))
    if not image_urls or len(image_urls) > 10:
        flash("Enter between 1 and 10 image URLs.", "warning")
        return redirect(url_for("scan"))

    category_folder = os.path.join(DATASET_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)
    existing_hashes = set()
    for path in os.listdir(category_folder):
        file_path = os.path.join(category_folder, path)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                existing_hashes.add(hashlib.sha256(file.read()).hexdigest())

    saved = 0
    rejected = 0
    for image_url in image_urls:
        try:
            image_bytes, _ = download_remote_image(image_url)
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            if image_hash in existing_hashes:
                rejected += 1
                continue
            filename = f"internet_{image_hash[:16]}.jpg"
            image_path = os.path.join(category_folder, filename)
            with open(image_path, "wb") as file:
                file.write(image_bytes)
            existing_hashes.add(image_hash)
            saved += 1
        except Exception:
            rejected += 1

    flash(f"Added {saved} new internet pictures to {category}; skipped {rejected} invalid or duplicate URL(s).", "success" if saved else "warning")
    return redirect(url_for("scan"))

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files.get("image")
    is_webcam_capture = bool(image and image.filename.startswith("webcam-capture"))

    if not image or not image.filename:
        flash("Please select an image.", "warning")
        return redirect(url_for("scan"))

    if not allowed_file(image.filename):
        flash("Only JPG, JPEG, and PNG images are allowed.", "warning")
        return redirect(url_for("scan"))

    extension = image.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid4().hex}.{extension}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(image_path)

    try:
        category, confidence = predict_waste(image_path)
    except Exception as error:
        flash(f"Prediction error: {error}", "danger")
        return redirect(url_for("scan"))

    if is_webcam_capture:
        dataset_folder = os.path.join(DATASET_FOLDER, category)
        os.makedirs(dataset_folder, exist_ok=True)
        dataset_filename = f"webcam_{uuid4().hex}.{extension}"
        shutil.copyfile(image_path, os.path.join(dataset_folder, dataset_filename))

    return render_template(
        "scan_result.html",
        prediction=category,
        confidence=round(confidence * 100, 2),
        image=filename,
        dataset_saved=is_webcam_capture
    )

@app.route("/report", methods=["GET", "POST"])
def report_waste():
    if request.method == "POST":
        waste_type = request.form.get("waste_type", "").strip()
        location = request.form.get("location", "").strip()
        description = request.form.get("description", "").strip()
        image = request.files.get("image")

        if not waste_type or not location:
            flash("Waste type and location are required.", "warning")
            return redirect(url_for("report_waste"))

        filename = None
        if image and image.filename:
            if not allowed_file(image.filename):
                flash("Report image must be JPG, JPEG, or PNG.", "warning")
                return redirect(url_for("report_waste"))
            extension = image.filename.rsplit(".", 1)[1].lower()
            filename = f"{uuid4().hex}.{extension}"
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO waste_reports "
            "(waste_type, location, description, image) "
            "VALUES (%s, %s, %s, %s)",
            (waste_type, location, description, filename)
        )
        mysql.connection.commit()
        cursor.close()

        flash("Environmental report submitted successfully.", "success")
        return redirect(url_for("report_waste"))

    return render_template("report.html")

@app.route("/admin/reports")
def admin_reports():
    cursor = mysql.connection.cursor()
    cursor.execute(
        "SELECT id, waste_type, location, description, image, "
        "report_date, status FROM waste_reports ORDER BY report_date DESC"
    )
    reports = cursor.fetchall()
    cursor.close()
    return render_template("admin_reports.html", reports=reports)

@app.route("/admin/stats")
def admin_stats():
    cursor = mysql.connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM waste_reports")
    total_reports = cursor.fetchone()[0]

    cursor.execute(
        "SELECT waste_type, COUNT(*) FROM waste_reports "
        "GROUP BY waste_type ORDER BY COUNT(*) DESC"
    )
    waste_stats = cursor.fetchall()

    cursor.execute(
        "SELECT location, COUNT(*) FROM waste_reports "
        "GROUP BY location ORDER BY COUNT(*) DESC"
    )
    location_stats = cursor.fetchall()

    cursor.close()
    return render_template(
        "admin_stats.html",
        total_reports=total_reports,
        waste_stats=waste_stats,
        location_stats=location_stats
    )

if __name__ == "__main__":
    threading.Timer(
        1.25,
        lambda: webbrowser.open_new("http://127.0.0.1:5000/scan")
    ).start()
    app.run(debug=True, use_reloader=False)
