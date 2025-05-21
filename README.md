# Wood Surface Anomaly Detection and Segmentation

This project aims to detect and segment anomalies on wood surfaces using unsupervised deep learning models. It includes a FastAPI-based backend and a React-based frontend. The following instructions will help you set up and run the project locally.

---

## 📁 Clone the Repository

```bash
git clone https://github.com/melisguclu/anomaly-detection.git
cd wood-anomaly-detection
```
---

## 🧪 Backend Setup (FastAPI)

### Step 1: Download Pretrained Model

Download the pretrained PaDiM model manually from the following Google Drive link:

🔗 [Download PaDiM model](https://drive.google.com/file/d/1iXv5bW00XWEBTJUeE8tvUTg2POZW-8Tx/view?usp=sharing)

After downloading, unzip the file and place it into the following directory inside the project:

```
models/padim/train_wood.pkl
```

Do not unzip it yet. Further steps will handle that automatically if necessary.

### Step 2: Set Up Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # For Unix/MacOS
venv\Scripts\activate    # For Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Step 4: Run the Backend Server

Navigate into the backend folder and start the FastAPI development server:

```bash
cd backend
uvicorn main:app --reload
```

By default, the API will be available at:

```
http://127.0.0.1:8000
```

You can view the automatically generated API docs at:

```
http://127.0.0.1:8000/docs
```

---

## 🚀 Frontend Setup (React)

### Step 1: Install Node Modules

Navigate to the frontend directory and install the dependencies:

```bash
cd ../frontend
npm install
```

### Step 2: Run the Development Server

```bash
npm run dev
```

This will start the React development server, typically available at:

```
http://localhost:5173
```

Make sure the backend is running simultaneously for the application to function properly.

---

## 🔗 Dataset Structure

Ensure the dataset is placed under the following structure:

```
wood_dataset/
└── wood/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── defect/
    └── ground_truth/
        └── defect/
```

This dataset structure is crucial for training and evaluating the models.

---

## 🌟 Notes

* Ensure all Python and Node dependencies are correctly installed.
* The pretrained model must be manually downloaded and placed into the correct folder.
* Backend and frontend must run concurrently for proper functionality.

---

## ✉️ Contact

For any issues or questions, please open an issue on the GitHub repository or contact the contributors.
