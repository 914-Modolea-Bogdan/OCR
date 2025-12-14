import hashlib
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import pandas as pd

from ocr_utils import extract_fields
from validation import validate_fields
from ai_corrector import ai_correct_fields


app = Flask(__name__, static_folder="static")
CORS(app)

@app.get("/")
def home():
    return app.send_static_file("index.html")

@app.route("/api/process", methods=["POST"])
def process_certificate():
    """Process uploaded certificate and return OCR results."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # Progress tracking
    progress_data = {"fraction": 0.0, "message": ""}
    
    def report_progress(fraction: float, message: str) -> None:
        progress_data["fraction"] = max(0, min(1.0, fraction))
        progress_data["message"] = message
    
    try:
        preview_image, extracted = extract_fields(
            io.BytesIO(file_bytes),
            preview=True,
            progress_callback=None,
            with_confidence=True,
        )


        # dict cu value-only (pentru Excel + rezultate simple)
        flat_results = {
            name: (v.get("value") if isinstance(v, dict) else v)
            for name, v in extracted.items()
        }

        corrected = ai_correct_fields(extracted)
        # validare se poate face acum fie pe flat_results, fie direct pe extracted,
        # pentru că validate_fields știe să le desfacă pe ambele.
        validation_issues = validate_fields(corrected)
        
        # Convert preview image to base64
        success, preview_encoded = cv2.imencode(".png", preview_image)
        if success:
            preview_base64 = base64.b64encode(preview_encoded.tobytes()).decode("utf-8")
        else:
            preview_base64 = None
        
        # Create Excel file
        results_df = pd.DataFrame([flat_results])
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="OCR")
        excel_buffer.seek(0)
        excel_bytes = excel_buffer.getvalue()
        excel_base64 = base64.b64encode(excel_bytes).decode("utf-8")
        
        # Convert original image to base64 for storage
        original_base64 = base64.b64encode(file_bytes).decode("utf-8")
        
        timestamp = datetime.now().isoformat(timespec="seconds")
        
        return jsonify({
            "success": True,
            "filename": file.filename or "certificate.png",
            "timestamp": timestamp,
            "file_hash": file_hash,
            "results": flat_results,  # compatibil cu ce aveai înainte
            "results_with_confidence": extracted,  # pentru benchmark și UI
            "validation_issues": validation_issues,
            "preview_image": preview_base64,
            "excel_file": excel_base64,
            "original_image": original_base64,
            "mime_type": file.content_type or "application/octet-stream",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

