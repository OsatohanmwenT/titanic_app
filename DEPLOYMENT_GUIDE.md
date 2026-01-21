# 🚀 Deployment Guide - Titanic Survival Prediction System

This guide provides step-by-step instructions for deploying your Titanic Survival Prediction System to various cloud platforms.

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:
- ✅ Trained model files in the `model/` directory
- ✅ All dependencies listed in `requirements.txt`
- ✅ Tested the application locally
- ✅ GitHub repository created and code pushed
- ✅ Updated personal information in submission files

## 🧪 Local Testing

Before deploying, test the application locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (if not already done)
python model/model_development.py

# 3. Run the Flask application
python app.py

# 4. Open browser and navigate to:
# http://localhost:5000
```

Test the prediction with sample data:
- **Test Case 1**: 1st class female, age 29, fare £100, embarked C → Should predict **Survived**
- **Test Case 2**: 3rd class male, age 25, fare £8, embarked S → Should predict **Did Not Survive**

---

## 🌐 Deployment Options

### Option 1: Render.com (Recommended) ⭐

**Advantages**: Free tier, automatic deployments, easy setup, supports Python

#### Steps:

1. **Prepare Your Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Titanic Survival Prediction System"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Sign Up for Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub account

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the Titanic project repository

4. **Configure Service**
   - **Name**: `titanic-survival-yourname`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Your app will be live at: `https://titanic-survival-yourname.onrender.com`

6. **Update Submission File**
   - Copy the live URL
   - Update `Titanic_hosted_webGUI_link.txt` with the URL

---

### Option 2: PythonAnywhere

**Advantages**: Free tier, Python-focused, easy for beginners

#### Steps:

1. **Sign Up**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com)
   - Create a free account

2. **Upload Code**
   - Go to "Files" tab
   - Create directory: `/home/yourusername/titanic`
   - Upload all project files

3. **Install Dependencies**
   - Go to "Consoles" tab
   - Start a Bash console
   ```bash
   cd ~/titanic
   pip install --user -r requirements.txt
   python model/model_development.py
   ```

4. **Configure Web App**
   - Go to "Web" tab
   - Click "Add a new web app"
   - Choose "Manual configuration"
   - Select Python 3.10

5. **Set Up WSGI File**
   - Edit the WSGI configuration file:
   ```python
   import sys
   path = '/home/yourusername/titanic'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

6. **Configure Static Files**
   - URL: `/static/`
   - Directory: `/home/yourusername/titanic/static/`

7. **Reload and Test**
   - Click "Reload" button
   - Visit: `https://yourusername.pythonanywhere.com`

---

### Option 3: Streamlit Cloud

**Note**: This requires converting the Flask app to Streamlit. Here's a quick Streamlit version:

#### Create `streamlit_app.py`:

```python
import streamlit as st
import joblib
import numpy as np
import os

# Page config
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('model/titanic_survival_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    le_sex = joblib.load('model/label_encoder_sex.pkl')
    le_embarked = joblib.load('model/label_encoder_embarked.pkl')
    return model, scaler, le_sex, le_embarked

model, scaler, le_sex, le_embarked = load_model()

# Header
st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict passenger survival using Machine Learning")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                             format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
        sex = st.radio("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=29)
    
    with col2:
        fare = st.number_input("Fare (£)", min_value=0.0, value=32.0, step=0.1)
        embarked = st.selectbox("Port of Embarkation", 
                               ["C", "Q", "S"],
                               format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])
    
    submitted = st.form_submit_button("Predict Survival", use_container_width=True)

if submitted:
    # Encode and predict
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]
    features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Display results
    if prediction == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
    
    col1, col2 = st.columns(2)
    col1.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    col2.metric("Death Probability", f"{probability[0]*100:.1f}%")
```

#### Deploy to Streamlit Cloud:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository and `streamlit_app.py`
6. Click "Deploy"

---

### Option 4: Vercel (with Serverless Functions)

Vercel is optimized for Next.js, but can host Flask apps with some configuration.

#### Steps:

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Create `vercel.json`**
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```

3. **Deploy**
   ```bash
   vercel
   ```

---

## 🔧 Troubleshooting

### Common Issues:

1. **Module Not Found Error**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **Model File Not Found**
   - Verify model files are in the `model/` directory
   - Ensure Git LFS is used for large files (if needed)

3. **Port Already in Use (Local)**
   - Change port in `app.py`: `app.run(port=5001)`

4. **Application Timeout**
   - Increase timeout in deployment settings
   - Optimize model loading with caching

### Model Files Too Large for Git?

If model files exceed GitHub's 100MB limit:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "model/*.pkl"
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

## 📝 Post-Deployment

After successful deployment:

1. **Test the Live Application**
   - Visit the deployed URL
   - Test with multiple passenger profiles
   - Verify predictions are working correctly

2. **Update Submission File**
   - Edit `Titanic_hosted_webGUI_link.txt`
   - Add your name, matric number, and live URL
   - Add GitHub repository link

3. **Final Checks**
   - ✅ Application loads without errors
   - ✅ All form inputs work correctly
   - ✅ Predictions display properly
   - ✅ UI is responsive on mobile
   - ✅ GitHub repository is public and accessible

4. **Submit to Scorac.com**
   - Ensure all files are in the correct structure
   - Upload before the deadline (February 5, 2026, 11:59 PM)

---

## 🎯 Submission Checklist

Before submitting, verify:

- [ ] All code files are present and organized correctly
- [ ] Model achieves at least 75% accuracy
- [ ] Web application runs without errors
- [ ] Application is deployed and accessible online
- [ ] `Titanic_hosted_webGUI_link.txt` is updated with all information
- [ ] GitHub repository is public
- [ ] README.md is comprehensive
- [ ] requirements.txt includes all dependencies
- [ ] Code is well-commented and clean

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review deployment platform documentation
3. Verify all files are correctly uploaded
4. Check application logs for error messages

---

**Good luck with your deployment! 🚀**
