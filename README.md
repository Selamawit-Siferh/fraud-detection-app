# fraud-detection-app

This project is a Machine Learning web application built using Streamlit for anomaly and suspicious transaction detection.

The application allows users to:

* Upload transaction datasets
* Detect suspicious transactions
* Visualize anomalies
* Run predictions using trained ML models

---

# Project Setup and Running the Streamlit Application (Windows)

This guide explains how to set up the project locally and run the Streamlit application on a Windows machine.

---

## 1. Install Python

Download and install Python from the official website:

https://www.python.org/downloads/windows/

Recommended version:

* Python 3.9 or newer

### Important

During installation, make sure to check:

Add Python to PATH

This allows Python commands to work from any terminal window.

---

## 2. Create a Virtual Environment (Recommended)

Using a virtual environment helps isolate project dependencies and prevents conflicts with other Python projects.

### Step 1: Open Command Prompt or PowerShell

### Step 2: Navigate to the Project Folder

Example:

```bash
cd C:\Users\Selamawitsiferh\Desktop\fraud-detection-app
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

```bash
.\venv\Scripts\activate
```

When activated, your terminal will display:

```bash
(venv)
```

---

## 3. Install Dependencies

Install all required Python libraries inside the activated virtual environment.

### Required Packages

```bash
pip install streamlit pandas numpy
pip install scikit-learn==1.6.1
```

### Optional Packages

```bash
pip install matplotlib seaborn joblib
```

### Why `scikit-learn==1.6.1`?

Using the same Scikit-learn version as the training environment prevents compatibility errors such as:

```text
AttributeError: _RemainderColsList
```

---

## 4. Create `requirements.txt`

Create a file named:

```text
requirements.txt
```

Add the following:

```txt
streamlit
pandas
numpy
scikit-learn==1.6.1
matplotlib
seaborn
joblib
```

This file is required for deployment on Streamlit Cloud.

---

## 5. Run the Streamlit Application

With the virtual environment activated, run:

```bash
streamlit run app.py
```

This command will:

* Start the Streamlit server
* Open the application in your default browser

Default local URL:

```text
http://localhost:8501
```

If the browser does not open automatically, copy and paste the URL into your browser.

---

## 6. Project Structure

```text
ML Assignment 2/
│── dbscan_model.pkl
├── app.py
├── kmeans_model.pkl
├── requirements.txt
├── README.md
├── Trx_ML.csv
└── Machine_Learning_Project.ipynb
└── reports.ipynb


```

---

# Uploading the Project to GitHub

## 1. Create a GitHub Repository

Create a repository on GitHub.

Example repository name:

```text
streamlit-anomaly-detection
```

---

## 2. Initialize Git

Inside the project folder:

```bash
git init
```

---

## 3. Add Files

```bash
git add .
```

---

## 4. Commit Files

```bash
git commit -m "Initial commit"
```

---

## 5. Connect the GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/Selamawit-Siferh/fraud-detection-app
```

---

## 6. Push the Project to GitHub

```bash
git branch -M main
git push -u origin main
```

The project is now uploaded to GitHub.

---

# Deploying the Application on Streamlit Cloud

Go to:

https://share.streamlit.io/

## Deployment Steps

1. Login with GitHub
2. Click "Create App"
3. Select:

   * Repository name
   * Branch: `main`
   * Main file path: `app.py`
4. Click "Deploy"

After deployment, Streamlit generates a public application URL such as:

```text
https://suspicious-transaction-detection.streamlit.app/
```

---

# Common Issues

## Module Not Found Error

Update dependencies:

```bash
pip freeze > requirements.txt
```

Push the updated file to GitHub.

---

## Git Push Error

Run:

```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

---

## Streamlit App Not Loading

Check:

* `requirements.txt` exists
* `app.py` is in the root folder
* model files are uploaded correctly

---

# Useful Commands

| Command                   | Description              |
| ------------------------- | ------------------------ |
| `streamlit run app.py`    | Run the app locally      |
| `git add .`               | Add files to Git         |
| `git commit -m "message"` | Commit changes           |
| `git push`                | Upload project to GitHub |

---

# Useful Links

* Streamlit Documentation: https://docs.streamlit.io
* GitHub: https://github.com
* Python Downloads: https://www.python.org/downloads/windows/
* Streamlit Community Cloud: https://share.streamlit.io
