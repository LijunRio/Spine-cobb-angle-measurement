
# Spine Cobb Angle Measurement

Automatic and semi-automatic **Cobb angle measurement** tool for spinal X-ray images (coronal & sagittal planes), implemented with **OpenCV + NumPy**.  
This project demonstrates how to estimate scoliosis and kyphosis/lordosis angles using image processing techniques.

---

## 🩻 Overview

**Cobb angle** is the gold standard to quantify spinal curvature (scoliosis or kyphosis).  
This project provides two measurement scripts:

| Script | Plane | Purpose |
|---------|--------|----------|
| `coronal_plane_cobb.py` | Coronal (frontal) | Measure scoliosis (lateral curvature) |
| `sagittal_plane_cobb.py` | Sagittal (side) | Measure kyphosis / lordosis (curvature in profile view) |

Each script:
1. Reads a spinal X-ray image  
2. Enhances contrast & detects edges  
3. Uses **Hough Line Transform** to find vertebral endplates  
4. Selects upper and lower end vertebrae lines  
5. Calculates the **Cobb angle** as the intersection angle of the two lines  
6. Visualizes the result on the image  

---

## 🧩 Algorithm Pipeline

```text
Input X-ray
   ↓
Grayscale conversion & noise filtering
   ↓
Edge detection (Canny) or threshold segmentation
   ↓
Morphological cleaning (optional)
   ↓
Hough Line Transform → detect candidate lines
   ↓
Select upper & lower endplate lines
   ↓
Compute Cobb angle between them
   ↓
Draw results and display/save
````


## 🧱 Project Structure

```
Spine-cobb-angle-measurement/
├── coronal_plane_cobb.py        # Coronal plane Cobb angle
├── sagittal_plane_cobb.py       # Sagittal plane Cobb angle
├── img/                         # Example X-ray images
├── test/ / test2/               # Additional test sets
└── README.md                    # You are here
```

---

## ⚙️ Requirements

Install dependencies via pip:

```bash
pip install opencv-python numpy matplotlib
```

Optional:

```bash
pip install tqdm
```

---

## 🚀 Usage

### 1. Coronal Plane (Scoliosis)

```bash
python coronal_plane_cobb.py --img img/example_ap.png
```

### 2. Sagittal Plane (Kyphosis/Lordosis)

```bash
python sagittal_plane_cobb.py --img img/example_lat.png
```

Each script will:

* Display the processed image with two detected lines
* Print the Cobb angle in degrees
* Optionally save a copy of the result image (you can enable this inside the script)

---


