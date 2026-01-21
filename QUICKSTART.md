# 🚀 Quick Start Guide

## Get Started in 3 Minutes!

### Step 1: Install Dependencies (30 seconds)
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (1 minute)
```bash
python model/model_development.py
```

You should see:
```
✓ Model saved to: model/titanic_survival_model.pkl
✓ Testing Accuracy: ~82%
```

### Step 3: Run the Application (30 seconds)
```bash
python app.py
```

### Step 4: Open in Browser
Navigate to: **http://localhost:5000**

---

## Test the Application

Try these test cases:

### Test 1: High Survival Probability
- **Passenger Class**: 1st Class
- **Gender**: Female
- **Age**: 29
- **Fare**: £100
- **Port**: Cherbourg (C)
- **Expected**: ✅ Survived (~98% probability)

### Test 2: Low Survival Probability
- **Passenger Class**: 3rd Class
- **Gender**: Male
- **Age**: 25
- **Fare**: £8
- **Port**: Southampton (S)
- **Expected**: ❌ Did Not Survive (~15% probability)

---

## What's Next?

1. ✅ Application works locally
2. 📤 Push to GitHub
3. 🌐 Deploy to Render.com (see DEPLOYMENT_GUIDE.md)
4. 📝 Update Titanic_hosted_webGUI_link.txt
5. 🎯 Submit to Scorac.com

---

## Need Help?

- **Deployment**: See `DEPLOYMENT_GUIDE.md`
- **Documentation**: See `README.md`
- **Project Details**: See walkthrough artifact

**Deadline**: February 5, 2026, 11:59 PM
