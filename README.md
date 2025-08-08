# COVID-19 CT Scan & Symptom-Based Severity Predictor

**COVID-19Predictor** is a deep learning–powered tool that predicts the likelihood of COVID-19 infection from **CT lung scans** and estimates the **severity level** by combining **patient symptoms** with the model’s prediction confidence via a **fuzzy logic controller**.
It also includes an **Explainable AI** (Grad-CAM) component to visually highlight the lung regions most influential in the model’s decision.

> Developed as a **course project** and **personal learning initiative**.

---

## Features

* **CT Scan Analysis**
  Utilizes a fine-tuned **ResNet-50** model (TensorFlow) to detect COVID-19 from CT images.

* **Symptom-Based Risk Assessment**
  Considers multiple clinical and demographic factors:

  ```
  'sex', 'patient_type', 'intubated', 'pneumonia', 'age', 'pregnancy',
  'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension',
  'other_disease', 'cardiovascular', 'obesity', 'renal_chronic',
  'tobacco', 'contact_other_covid', 'icu'
  ```

* **Fuzzy Logic Risk Fusion**

  * Symptom-based risk: **high\_risk**, **low\_risk**
  * CT scan severity: **low**, **moderate**, **high**
  * Combined output: **low**, **medium**, **high**

* **Custom Preprocessing**
  Includes a **custom binary lung extraction filter** to isolate lung regions before prediction.

* **Explainable AI with Grad-CAM**
  Superimposes a heatmap on CT images to highlight areas most responsible for the COVID-19 prediction.

---

## Dataset

This project uses the [COVID-CTset](https://github.com/mr7495/COVID-CTset) dataset, which contains:

* 349 positive COVID-19 CT scans from 216 patients.
* 397 negative CT scans from 397 patients.

Please cite the dataset source if you use it in your work.

---

## Sample Outputs

**Prediction with Grad-CAM Heatmap**
![Grad-CAM Example](../classifier/test_images/cov_gradcam.png)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/Covid19Predictor.git
   cd Covid19Predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and place dataset**
   Place the COVID-CTset dataset in the `data/` directory (or update dataset path in the config).

---

## Usage

Run the predictor on a CT scan image and patient data:

```bash
python predict.py --image path/to/ct_scan.png --symptoms path/to/symptoms.csv
```

* **`--image`**: Path to CT scan image file.
* **`--symptoms`**: Path to CSV containing symptom data for the patient.

Output:

* COVID-19 prediction probability.
* Symptom-based risk level.
* Combined fuzzy logic–based severity prediction.
* Grad-CAM heatmap visualization.

---

## Dependencies

* **TensorFlow** – Deep learning model (ResNet-50)
* **NumPy** – Numerical computations
* **OpenCV** – Image preprocessing and lung segmentation
* **Matplotlib** – Visualization
* **scikit-fuzzy** – Fuzzy logic inference system

Install all dependencies:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-fuzzy
```

---

## License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## References

* COVID-CTset Dataset: [https://github.com/mr7495/COVID-CTset](https://github.com/mr7495/COVID-CTset)
* Grad-CAM: Selvaraju, R. R., et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
* scikit-fuzzy: [https://pythonhosted.org/scikit-fuzzy/](https://pythonhosted.org/scikit-fuzzy/)