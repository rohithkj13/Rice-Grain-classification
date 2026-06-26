# Rice Classification Project

Flask-based rice grain classification app for these five classes:
- `Basmati`
- `Arborio`
- `Jasmine`
- `Karacadag`
- `Ipsala`

The current project is centered on:
- [flask_app.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/flask_app.py)
- [src/hybrid_classifier.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/hybrid_classifier.py)
- [src/stable_predictor.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/stable_predictor.py)
- [static/js/app.js](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/static/js/app.js)
- [templates/index.html](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/templates/index.html)

## Run

```powershell
cd "c:\Users\ROHITH\OneDrive\Desktop\rg project\Rice_Classification_Project"
python flask_app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Main Modes

Normal app flow:
- single-grain extraction
- AI / corrected morphology / hybrid prediction
- stability layer for repeatable output

Deterministic analyst mode:
- enabled from the frontend request flow
- extracts fixed visual features
- scores all five classes out of `5`
- returns:
  - `Features`
  - `Scores`
  - `Final Answer`

## API

### Health

```http
GET /api/health
```

### Predict

```http
POST /api/predict
```

Accepted inputs:
- multipart file upload
- base64 image data
- image URL

Accepted formats:
- `image/jpeg`
- `image/png`
- `image/webp`

### Deterministic Analyst Output

When `analyst_mode=true`, the backend returns a deterministic scoring report like:

```json
{
  "Features": {
    "length_category": "long",
    "shape": "slender",
    "edge": "pointed",
    "color": "translucent",
    "thickness": "thin"
  },
  "Scores": {
    "basmati": "5/5",
    "arborio": "1/5",
    "jasmine": "2/5",
    "karacadag": "1/5",
    "ipsala": "0/5"
  },
  "Final Answer": {
    "Predicted Class": "basmati",
    "Confidence": "100%",
    "Reason": "Selected by deterministic feature scoring."
  }
}
```

### Stable Output

The backend also supports stable repeatable prediction with image hashing, caching, and locked output for repeated images.

## Project Structure

```text
Rice_Classification_Project/
├── flask_app.py
├── app.py
├── main.py
├── config.py
├── README.md
├── requirements.txt
├── requirements_flask.txt
├── train_knn_model.py
├── train_rice_model.py
├── train_rice_quick.py
├── diagnose_integrated.py
├── models/
├── Rice_Image_Dataset/
├── src/
├── static/
├── templates/
└── utils/
```

## Important `src` Files

- [advanced_integrated_classifier.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/advanced_integrated_classifier.py)
- [advanced_multi_grain.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/advanced_multi_grain.py)
- [corrected_rice_classifier.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/corrected_rice_classifier.py)
- [hybrid_classifier.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/hybrid_classifier.py)
- [knn_classifier.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/knn_classifier.py)
- [optimized_ai_classifier.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/optimized_ai_classifier.py)
- [predict.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/predict.py)
- [preprocess.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/preprocess.py)
- [single_grain_analyzer.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/single_grain_analyzer.py)
- [stable_predictor.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/stable_predictor.py)
- [train_model.py](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/src/train_model.py)

## Training

Train the main image model:

```powershell
python train_rice_model.py
```

Train the KNN feature model:

```powershell
python train_knn_model.py
```

Quick training path:

```powershell
python train_rice_quick.py
```

## Verification

Python syntax check:

```powershell
python -m py_compile flask_app.py app.py main.py diagnose_integrated.py
```

## Notes

- The repo has been cleaned to keep the active app and core classifiers only.
- The frontend now sends deterministic analyst mode requests by default.
- The dataset remains under [Rice_Image_Dataset](/c:/Users/ROHITH/OneDrive/Desktop/rg%20project/Rice_Classification_Project/Rice_Image_Dataset).
