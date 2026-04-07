import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import models, transforms
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "final-batch-test-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# EXACT list matching your 5 files
ALL_MODELS = [
    "AlexNet", 
    "ConvNeXt", 
    "EfficientNet", 
    "ShuffleNet", 
    "SqueezeNet"
]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Standard Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- Model Loader --------------------
def load_model_and_layer(model_name):
    """
    Loads one of the 5 specific models.
    """
    map_loc = "cpu"
    
    # 1. Construct filename: "AlexNet" -> "alexnet_clinical.pt"
    file_prefix = model_name.lower()
    weight_path = os.path.join(MODEL_DIR, f"{file_prefix}_clinical.pt")
    
    logger.debug(f"Looking for weights at: {weight_path}")
    
    if not os.path.exists(weight_path):
        logger.error(f"❌ Weights not found: {weight_path}")
        return None, None

    target_layer = None
    model = None

    # 2. Define Architecture
    try:
        if model_name == "AlexNet":
            model = models.alexnet(weights=None)
            # AlexNet's final layer is classifier[6]
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
            target_layer = model.features[-1]

        elif model_name == "ConvNeXt":
            # Using convnext_tiny. If your weights are 'small' or 'base', change this line.
            model = models.convnext_tiny(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
            target_layer = model.features[-1]

        elif model_name == "EfficientNet":
            # Using efficientnet_v2_s. If you used b0/b1, change this line.
            model = models.efficientnet_v2_s(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            target_layer = model.features[-1]

        elif model_name == "ShuffleNet":
            model = models.shufflenet_v2_x1_0(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            target_layer = model.conv5

        elif model_name == "SqueezeNet":
            model = models.squeezenet1_0(weights=None)
            # SqueezeNet uses Conv2d as classifier
            model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1))
            model.num_classes = 2
            target_layer = model.features[-1]

        # 3. Load Weights
        try:
            # First try with weights_only parameter
            state_dict = torch.load(weight_path, map_location=map_loc, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Successfully loaded {model_name} with weights_only=True")
        except TypeError:
            # Fall back for older PyTorch versions
            state_dict = torch.load(weight_path, map_location=map_loc)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Successfully loaded {model_name} without weights_only")
        except RuntimeError as e:
            logger.error(f"❌ Architecture mismatch for {model_name}. Error: {e}")
            return None, None
            
        model.eval()
        return model, target_layer

    except Exception as e:
        logger.error(f"❌ Error setting up {model_name}: {e}")
        return None, None

# -------------------- Grad-CAM Generator --------------------
def generate_gradcam(model, target_layer, input_tensor, original_img_np, class_idx):
    if target_layer is None:
        return original_img_np

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    # Register hooks
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook) 

    # Zero grad and run backward
    model.zero_grad()
    input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
    
    output = model(input_tensor_grad)
    score = output[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    if not gradients or not activations:
        logger.warning("No gradients or activations captured for Grad-CAM")
        return original_img_np

    grads = gradients[0][0]
    acts = activations[0][0]

    # Weighted sum
    pooled_grads = grads.mean(dim=(1, 2), keepdim=True)
    cam = (pooled_grads * acts).sum(dim=0)
    cam = torch.relu(cam)
    cam_np = cam.cpu().numpy()
    
    # Normalize & Colorize
    cam_resized = cv2.resize(cam_np, (224, 224))
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    
    if cam_max > cam_min:
        cam_normalized = (cam_resized - cam_min) / (cam_max - cam_min)
    else:
        cam_normalized = np.zeros_like(cam_resized)
        
    heatmap = np.uint8(255 * cam_normalized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    superimposed = cv2.addWeighted(original_img_np, 0.6, heatmap_rgb, 0.4, 0)
    return superimposed

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("test.html", stats=None)

    logger.debug("POST request received")
    
    # 1. Handle Upload
    if 'images' not in request.files:
        logger.error("No files part in request")
        flash("No files selected.", "danger")
        return redirect(url_for("index"))
    
    files = request.files.getlist("images")
    logger.debug(f"Number of files received: {len(files)}")
    
    # Filter out empty filenames
    files = [f for f in files if f and f.filename and f.filename.strip()]
    
    if not files:
        logger.error("No valid files selected")
        flash("No valid files selected.", "danger")
        return redirect(url_for("index"))

    # 2. Save Images
    saved_images = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Generate unique filename
                uid = uuid.uuid4().hex[:8]
                original_filename = secure_filename(file.filename)
                filename = f"{uid}_{original_filename}"
                
                # Save to uploads folder first
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                logger.debug(f"File saved to: {upload_path}")
                
                # Copy to static folder for serving
                static_path = os.path.join(STATIC_DIR, filename)
                
                # Open and process image to ensure it's valid
                with Image.open(upload_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Save to static folder
                    img.save(static_path)
                
                saved_images.append(static_path)
                logger.debug(f"Image processed and saved to static: {static_path}")
                
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                flash(f"Error processing {file.filename}: {str(e)}", "danger")
        else:
            logger.warning(f"Invalid file type: {file.filename}")
            flash(f"Invalid file type: {file.filename}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}", "warning")

    if not saved_images:
        logger.error("No images were successfully saved")
        flash("No valid images were uploaded.", "danger")
        return redirect(url_for("index"))

    logger.debug(f"Successfully saved {len(saved_images)} images")

    # 3. Process with all 5 models
    stats = {}

    for model_name in ALL_MODELS:
        try:
            logger.info(f"--- Running {model_name} ---")
            model, target_layer = load_model_and_layer(model_name)
            
            if model is None:
                logger.warning(f"Skipping {model_name} - model could not be loaded")
                continue

            model_stats = {
                "total_conf": 0.0,
                "count": 0,
                "cancer_cases": [],
                "normal_cases": []
            }

            for img_path in saved_images:
                try:
                    # Load Image
                    pil_img = Image.open(img_path).convert("RGB")
                    original_np = np.array(pil_img.resize((224, 224)))
                    input_tensor = transform(pil_img).unsqueeze(0)

                    # Predict
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                        pred_idx = int(torch.argmax(output, dim=1).cpu().numpy()[0])
                        confidence = float(probs[pred_idx] * 100.0)

                    logger.debug(f"{model_name} - Prediction: {pred_idx}, Confidence: {confidence:.2f}%")

                    # Grad-CAM
                    grad_img_np = original_np
                    if target_layer:
                        try:
                            grad_img_np = generate_gradcam(model, target_layer, input_tensor, original_np, pred_idx)
                        except Exception as e:
                            logger.error(f"GradCAM error for {model_name}: {e}")

                    # Save Output
                    base_name = os.path.basename(img_path)
                    cam_filename = f"grad_{model_name}_{base_name}"
                    cam_path = os.path.join(STATIC_DIR, cam_filename)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(cam_path), exist_ok=True)
                    
                    # Save the Grad-CAM image
                    cv2.imwrite(cam_path, cv2.cvtColor(grad_img_np, cv2.COLOR_RGB2BGR))
                    logger.debug(f"Grad-CAM saved to: {cam_path}")

                    # Data Entry
                    result_entry = {
                        "gradcam": f"/static/{cam_filename}",
                        "conf": round(confidence, 2)
                    }

                    model_stats["total_conf"] += confidence
                    model_stats["count"] += 1
                    
                    # 0 = Cancer, 1 = Normal (Standard for ImageFolder)
                    if pred_idx == 0:
                        model_stats["cancer_cases"].append(result_entry)
                    else:
                        model_stats["normal_cases"].append(result_entry)

                except Exception as e:
                    logger.error(f"Error processing image {img_path} with {model_name}: {e}")
                    continue

            # Averages
            if model_stats["count"] > 0:
                model_stats["avg"] = round(model_stats["total_conf"] / model_stats["count"], 2)
            else:
                model_stats["avg"] = 0.0

            stats[model_name] = model_stats
            logger.info(f"{model_name} completed. Average confidence: {model_stats['avg']}%")
            
            # Clean up GPU memory if using CUDA
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            continue

    # 4. Determine Winner
    best_model = "None"
    best_score = 0
    if stats:
        best_model = max(stats, key=lambda x: stats[x]["avg"])
        best_score = stats[best_model]["avg"]
        logger.info(f"Best model: {best_model} with score {best_score}%")

    # Clean up uploaded files (optional)
    for img_path in saved_images:
        try:
            # Remove from uploads folder if needed
            upload_filename = os.path.basename(img_path)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except Exception as e:
            logger.warning(f"Could not delete upload {upload_path}: {e}")

    return render_template("test.html", stats=stats, best_model=best_model, best_score=best_score)

# Add error handler for large files
@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum size is 16MB.", "danger")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(port=5001, debug=True, use_reloader=False)