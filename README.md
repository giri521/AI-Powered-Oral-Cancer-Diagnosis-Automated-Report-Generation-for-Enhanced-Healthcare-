# 🧠 AI-Powered Oral Cancer Diagnosis System

### Automated Report Generation for Enhanced Healthcare

---

## 📌 Project Overview

This project is an advanced AI-powered web application designed to detect oral cancer from clinical images using deep learning models. It not only predicts the presence of cancer but also generates a detailed diagnostic PDF report and sends it via email.

The system integrates explainable AI (Grad-CAM) to visualize model attention and improve interpretability for medical professionals.

---

## 🚀 Key Features

* 🔍 Deep Learning-based Oral Cancer Detection
* 📊 Probability-based Prediction (Cancer / Normal)
* 🔥 Grad-CAM Visualization (Explainable AI)
* 📄 Automated PDF Medical Report Generation
* 📧 Email Report Delivery System
* 🌐 Flask-based Web Application
* 🗄️ Secure Environment-based Configuration

---

## 🧠 How It Works

1. User uploads an oral image
2. Selected AI model analyzes the image
3. System predicts:

   * Cancer / Normal
   * Confidence score
4. Grad-CAM highlights affected regions
5. A detailed PDF report is generated
6. Report is optionally sent via email

---

## 🧬 AI Models Used

* EfficientNetV2
* ConvNeXt Tiny

(All models are implemented using PyTorch)

---

## 🛠️ Tech Stack

* **Backend:** Python (Flask)
* **Frontend:** HTML, CSS, JavaScript
* **AI Framework:** PyTorch, Torchvision
* **Image Processing:** OpenCV, PIL
* **Report Generation:** FPDF
* **Email Service:** SMTP (Gmail)
* **Other Libraries:** NumPy, dotenv

---

## 📁 Project Structure

```bash
├── app.py
├── model/
├── static/
├── templates/
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/giri521/AI-Powered-Oral-Cancer-Diagnosis-Automated-Report-Generation-for-Enhanced-Healthcare-.git
cd AI-Powered-Oral-Cancer-Diagnosis-Automated-Report-Generation-for-Enhanced-Healthcare-
pip install -r requirements.txt
python app.py
```

---

## 🔐 Environment Variables

Create a `.env` file:

```bash
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
FLASK_SECRET=your_secret_key
```

---

## 📸 Screenshots

(Add screenshots here)

* Upload Page
* Prediction Output
* Grad-CAM Visualization
* PDF Report

---

## ⚠️ Model Files Notice

Due to GitHub file size limitations, trained model files (.pt) are not included in this repository.

👉 Download models here: (Add Google Drive link)

👉 Place them inside:

```
model/
```

---

## 📊 Output Details

* Prediction: Cancer / Normal
* Confidence Score (%)
* Risk Level (Early / High)
* Grad-CAM Heatmap
* Auto-generated PDF Report

---

## 🎯 Applications

* Early Oral Cancer Detection
* Clinical Decision Support
* AI-assisted Medical Diagnosis
* Healthcare Automation

---

## 🔮 Future Scope

* Real-time camera detection
* Mobile application
* Cloud deployment
* Multi-class cancer detection

---

## 👨‍💻 Author

**Vennapusa Girivardhan Reddy**
AI & ML Enthusiast | Python Developer

---

## ⭐ Support

If you found this project useful, please ⭐ the repository on GitHub!
