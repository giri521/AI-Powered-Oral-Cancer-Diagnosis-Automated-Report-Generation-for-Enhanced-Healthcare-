import os
import uuid
import datetime
import smtplib
import ssl
from email.message import EmailMessage
from io import BytesIO # Not used, but kept for completeness
import numpy as np
import cv2
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
from fpdf import FPDF

import torch
import torch.nn as nn
from torchvision import models, transforms


# -------------------- Configuration --------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}

# Email config from environment variables
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")

# Use 'cuda' if available, otherwise default to 'cpu'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Flask app --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-in-prod")


# -------------------- Model utilities --------------------
def load_model_instance(model_choice: str, img_type: str):
    """
    Load the appropriate PyTorch model and return (model, target_layer).
    This function assumes model weights exist at model/{name}_{img_type}.pt
    """
    # map_loc is used for torch.load() to handle device mapping during weight loading
    map_loc = "cpu" if DEVICE.type == 'cpu' else None

    if model_choice == "EfficientNetV2":
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        weight_path = os.path.join(MODEL_DIR, f"efficientnet_{img_type}.pt")
        
        # Target layer for Grad-CAM
        target_layer = model.features[-1]
    else:  # ConvNeXt tiny
        model_choice = "ConvNeXt_Tiny" # Standardize name
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 2)
        weight_path = os.path.join(MODEL_DIR, f"convnext_{img_type}.pt")
        
        # Target layer for Grad-CAM
        # ConvNeXt has features organized in blocks; features[-1] is the last block
        target_layer = model.features[-1] 

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing weights: {weight_path}")
    
    # Load weights, map to CPU/GPU if needed
    model.load_state_dict(torch.load(weight_path, map_location=map_loc))
    
    # *** ROBUSTNESS FIX: Move the model to the target device (CPU or CUDA) ***
    model.to(DEVICE)
    
    model.eval()
    return model, target_layer


# global transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def generate_gradcam(model, target_layer, input_tensor, original_img_np: np.ndarray, class_idx):
    """
    Generate Grad-CAM and superimpose it on the original image.
    """
    gradients = []
    activations = []

    # Move input tensor to the correct device for inference
    input_tensor = input_tensor.to(DEVICE)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    # Register hooks
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    
    # Calculate gradient for the predicted class
    score = output[0, class_idx]
    
    # Perform backward pass
    score.backward(retain_graph=False)

    # Remove hooks
    h1.remove()
    h2.remove()

    # Get gradients and activations (now on the device)
    grads = gradients[0][0]
    acts = activations[0][0]

    # Grad-CAM calculation
    pooled_grads = grads.mean(dim=(1, 2), keepdim=True)
    cam = (pooled_grads * acts).sum(dim=0)
    cam = torch.relu(cam)
    
    # Move to CPU for numpy conversion
    cam_np = cam.cpu().numpy()
    
    # Image processing and visualization
    cam_resized = cv2.resize(cam_np, (224, 224))
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max > cam_min:
        cam_normalized = (cam_resized - cam_min) / (cam_max - cam_min)
    else:
        cam_normalized = np.zeros_like(cam_resized)
        
    heatmap = np.uint8(255 * cam_normalized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # original_img_np is assumed to be 224x224 RGB
    superimposed_img = cv2.addWeighted(original_img_np, 0.6, heatmap_rgb, 0.4, 0)
    
    return superimposed_img


# -------------------- Helper functions --------------------
def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return "." in filename and ext in ALLOWED_EXT


def save_numpy_image(img_np: np.ndarray, out_path: str):
    # PIL/Matplotlib uses RGB, OpenCV uses BGR. Convert to BGR for cv2.imwrite.
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


# -------------------- PDF Generation --------------------
def create_report_pdf(patient_info: dict, analysis_details: dict, uploaded_im_path: str, gradcam_path: str,
                      prediction: str, confidence: float, risk: str, out_pdf_path: str,
                      hospital_info: dict = None):
    """
    Create a detailed and formatted PDF report. (Original function kept as it is excellent)
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Header ---
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, hospital_info.get("name", "Kalasalingam Hospital"), ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    # The address is broken over multiple lines, using multi_cell for better formatting
    address_lines = hospital_info.get("address", "Department of Oral Pathology & Diagnostics,\nKalasalingam Academy of Research and Education,\nKrishnankoil, Tamil Nadu, India").split('\n')
    for line in address_lines:
        pdf.cell(0, 4, line, ln=True, align="C")
    
    pdf.cell(0, 6, f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)

    # --- Patient & Analysis Details Section ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient & Analysis Details", ln=True, align="L")
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "", 10)

    pdf.cell(95, 6, f"Patient Name: {patient_info.get('name','')}", border=1, ln=False, fill=True)
    pdf.cell(95, 6, f"Analysis ID: {patient_info.get('reference','-')}", border=1, ln=True, fill=True)
    pdf.cell(95, 6, f"Age / Sex: {patient_info.get('age','')} / {patient_info.get('sex','')}", border=1, ln=False)
    pdf.cell(95, 6, f"AI Model Used: {analysis_details.get('model_choice', 'N/A')}", border=1, ln=True)
    pdf.cell(95, 6, f"DOB: {patient_info.get('dob','')}", border=1, ln=False, fill=True)
    pdf.cell(95, 6, f"Original Filename: {analysis_details.get('original_filename', 'N/A')}", border=1, ln=True, fill=True)
    pdf.ln(5)

    # --- Clinical Information Section ---
    if patient_info.get("clinical_summary") or patient_info.get("causes"):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Clinical Information", ln=True, align="L")
        pdf.set_font("Arial", "", 10)
        if patient_info.get("clinical_summary"):
            pdf.multi_cell(0, 5, f"Summary: {patient_info.get('clinical_summary')}", border=1)
        if patient_info.get("causes"):
            pdf.multi_cell(0, 5, f"Reported History/Causes: {patient_info.get('causes')}", border=1)
        pdf.ln(5)

    # --- AI Diagnostic Results Section ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "AI Diagnostic Results", ln=True, align="L")
    
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Model Probability Scores:", ln=True)
    pdf.set_font("Arial", "", 10)
    probs = analysis_details.get("all_probs", [0, 0])
    # Assuming index 0 is Cancer and index 1 is Normal based on route logic
    cancer_prob = probs[0] * 100
    normal_prob = probs[1] * 100
    pdf.cell(95, 8, f"Cancer Probability: {cancer_prob:.2f}%", border=1, align="C", fill=True)
    pdf.cell(95, 8, f"Normal Probability: {normal_prob:.2f}%", border=1, align="C", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(60, 8, "Final Prediction:", align="L")
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(220, 53, 69) if prediction == "Cancer" else pdf.set_text_color(25, 135, 84)
    pdf.cell(40, 8, prediction, border=1, align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(40, 8, "Assessed Risk:", align="R")
    pdf.set_font("Arial", "", 10)
    pdf.cell(50, 8, risk, align="C", ln=True)
    pdf.ln(5)
    
    # --- Visual Evidence Section ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Visual Evidence", ln=True, align="L")
    
    y_start_images = pdf.get_y()
    pdf.set_font("Arial", "B", 10)
    pdf.cell(95, 6, "Submitted Clinical Image", align="C", ln=False)
    pdf.cell(95, 6, "AI Attention Map (Grad-CAM)", align="C", ln=True)
    
    # Note: fpdf's image function uses the path directly.
    # The image size must be specified in the PDF's unit (mm). w=85 mm is about 3.3 inches.
    try:
        pdf.image(uploaded_im_path, x=15, y=y_start_images + 8, w=85)
    except Exception as e:
        pdf.set_xy(15, y_start_images + 8)
        pdf.cell(85, 85, "Image Error", 1, align="C")
        # print(f"Error embedding uploaded image: {e}") # Keep printing for server-side debugging
    try:
        pdf.image(gradcam_path, x=110, y=y_start_images + 8, w=85)
    except Exception as e:
        pdf.set_xy(110, y_start_images + 8)
        pdf.cell(85, 85, "Image Error", 1, align="C")
        # print(f"Error embedding Grad-CAM image: {e}") # Keep printing for server-side debugging
    
    pdf.set_y(y_start_images + 100) # Move cursor down after images

    # --- Outcome & Recommendations Section ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Outcome & Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)
    outcome_text = patient_info.get("outcome_text", "")
    if not outcome_text:
        if prediction.lower() == "normal":
            outcome_text = "The AI model did not detect features consistent with malignancy. Clinical correlation and routine follow-up are recommended."
        else:
            outcome_text = ("The AI model identified features suspicious for epithelial dysplasia or malignancy. This is a preliminary finding and requires histopathological confirmation. Urgent referral to a specialist for biopsy is strongly recommended.")
    pdf.multi_cell(0, 6, outcome_text, border=1)
    pdf.ln(10)

    # --- Signature & Disclaimer ---
    pdf.cell(0, 6, "Electronically Signed and Verified By:", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, hospital_info.get("pathologist", "Dr. Rania Younis, BDS, MDS, PhD"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, hospital_info.get("role", "Consultant Pathologist"), ln=True)
    
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(128, 128, 128)
    disclaimer = hospital_info.get("disclaimer", "DISCLAIMER: This is an AI-assisted report. It is intended for use by qualified healthcare professionals as a supplementary diagnostic tool and not as a standalone diagnosis. All findings must be correlated with clinical and histopathological results.")
    pdf.multi_cell(0, 4, disclaimer, align="C")

    pdf.output(out_pdf_path)


# -------------------- Email sending --------------------
def send_pdf_email(receiver_email: str, subject: str, body: str, attachment_path: str):
    # Existing excellent email function.
    if not (EMAIL_USER and EMAIL_PASS):
        raise ValueError("Email credentials not configured in environment variables.")

    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="application", subtype="pdf", filename=os.path.basename(attachment_path))

    # Standard SSL context
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # get form fields
    name = request.form.get("name", "").strip()
    age = request.form.get("age", "")
    sex = request.form.get("sex", "")
    dob = request.form.get("dob", "")
    email = request.form.get("email", "").strip()
    model_choice = request.form.get("model_choice", "EfficientNetV2")
    img_type = request.form.get("img_type", "clinical")
    clinical_summary = request.form.get("clinical_summary", "")
    causes = request.form.get("causes", "")

    file = request.files.get("image")
    if not file or file.filename == "":
        flash("No image selected. Please upload or capture an image.", "danger")
        return redirect(url_for("index"))

    original_filename = secure_filename(file.filename)
    if not allowed_file(original_filename):
        flash("File type not allowed. Please use JPG, JPEG, or PNG.", "danger")
        return redirect(url_for("index"))

    # save uploaded original image
    uid = uuid.uuid4().hex[:10]
    filename = f"{uid}_{original_filename}"
    upload_path = os.path.join(STATIC_DIR, filename)
    file.save(upload_path)

    # load model
    try:
        model, target_layer = load_model_instance(model_choice, img_type)
    except Exception as e:
        flash(f"Model loading error: {e}", "danger")
        # Cleanup uploaded file if model load fails
        os.remove(upload_path)
        return redirect(url_for("index"))

    # prepare input tensor and numpy image for grad-cam
    pil_img = Image.open(upload_path).convert("RGB")
    pil_img_resized = pil_img.resize((224, 224))
    original_img_for_cam = np.array(pil_img_resized)
    input_tensor = transform(pil_img).unsqueeze(0)

    # inference
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        # Classes: 0 is Cancer, 1 is Normal (based on typical ML output if trained with 0=Malignant, 1=Benign)
        predicted = int(torch.argmax(output, dim=1).cpu().numpy()[0])
        confidence = float(probs[predicted] * 100.0)

    classes = ["Cancer", "Normal"]
    prediction = classes[predicted]
    # Risk calculation remains the same
    risk = "None" if prediction == "Normal" else ("Early Stage" if confidence < 93 else "Highly Infected")

    # generate gradcam
    grad_filename = f"grad_{uid}.jpg"
    grad_path = os.path.join(STATIC_DIR, grad_filename)
    try:
        gradcam_rgb = generate_gradcam(model, target_layer, input_tensor, original_img_for_cam, predicted)
        save_numpy_image(gradcam_rgb, grad_path)
    except Exception as e:
        flash(f"Grad-CAM generation failed: {e}. Report generated without attention map.", "warning")
        # Create a placeholder image to avoid PDF generation crash if possible
        gradcam_rgb = np.zeros((224, 224, 3), dtype=np.uint8) 
        Image.fromarray(gradcam_rgb).save(grad_path) # Save a blank image
        

    # create PDF
    pdf_fname = f"{secure_filename(name.replace(' ', '_')) or 'patient'}_{uid}_report.pdf"
    pdf_path = os.path.join(STATIC_DIR, pdf_fname)

    patient_info = {
        "name": name, "age": age, "sex": sex, "dob": dob, "email": email,
        "reference": uid, "clinical_summary": clinical_summary, "causes": causes
    }
    
    analysis_details = {
        "model_choice": model_choice,
        "original_filename": original_filename,
        "all_probs": probs
    }

    hospital_info = {
        "name": "Kalasalingam Hospital",
        "address": (
            "Department of Oral Pathology & Diagnostics,\n"
            "Kalasalingam Academy of Research and Education,\n"
            "Krishnankoil, Tamil Nadu, India"
        ),
        "pathologist": "Dr. Girivardhan Reddy Vennapusa",
        "role": "AI Researcher & Consultant Pathologist",
        "disclaimer": (
            "This is an AI-assisted diagnostic report generated at Kalasalingam Hospital. "
            "It is intended for use by qualified healthcare professionals as a supplementary diagnostic tool "
            "and not as a standalone diagnosis. All findings must be confirmed with histopathology "
            "and clinical correlation."
        )
    }

    try:
        create_report_pdf(patient_info=patient_info,
                          analysis_details=analysis_details,
                          uploaded_im_path=upload_path,
                          gradcam_path=grad_path,
                          prediction=prediction,
                          confidence=confidence,
                          risk=risk,
                          out_pdf_path=pdf_path,
                          hospital_info=hospital_info)
    except Exception as e:
        flash(f"PDF generation failed: {e}", "danger")
        # Attempt to cleanup partial files
        if os.path.exists(upload_path): os.remove(upload_path)
        if os.path.exists(grad_path): os.remove(grad_path)
        return redirect(url_for("index"))

    # send email
    email_status = None
    if email:
        try:
            subject = "Your AI Oral Cancer Diagnostic Report"
            body = (f"Dear {name or 'Patient'},\n\n"
                    "Please find attached your AI-assisted Oral Cancer Diagnostic Report.\n\n"
                    "This report is intended for use by healthcare professionals.")
            send_pdf_email(receiver_email=email, subject=subject, body=body, attachment_path=pdf_path)
            email_status = "sent"
        except Exception as e:
            email_status = f"failed: {e}"

    # show results page
    result = {
        "prediction": prediction,
        "confidence": f"{confidence:.2f}",
        "risk": risk,
        "upload_image": f"/static/{filename}",
        "grad_image": f"/static/{grad_filename}",
        "pdf_file": f"/static/{pdf_fname}",
        "email_status": email_status,
        "email": email
    }

    return render_template("index.html", result=result)


@app.route("/download/<path:filename>")
def download(filename):
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        flash("File not found", "danger")
        return redirect(url_for("index"))
    
    # Send the file
    response = send_file(file_path, as_attachment=True)
    
    # NOTE: You might want to add a cleanup step here to delete the file
    # after it has been served/downloaded to prevent the static folder from filling up.
    
    return response


# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)