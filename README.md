# Titanic Survival Prediction System

A machine learning web application that predicts passenger survival on the Titanic using a Random Forest Classifier.

## 📋 Project Overview

This project implements a complete machine learning pipeline including:
- Data preprocessing and feature engineering
- Random Forest Classifier model training
- Modern web-based GUI for predictions
- Model persistence using Joblib
- Deployment-ready configuration

## 🎯 Features Used

The model uses the following 5 features to make predictions:
1. **Pclass** - Passenger Class (1st, 2nd, or 3rd)
2. **Sex** - Gender (Male/Female)
3. **Age** - Passenger Age
4. **Fare** - Ticket Fare
5. **Embarked** - Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Titanic_Project_yourName_matricNo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (if not already trained):
```bash
python model/model_development.py
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
Titanic_Project_yourName_matricNo/
│
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── Titanic_hosted_webGUI_link.txt # Deployment information
│
├── model/
│   ├── model_building.ipynb       # Jupyter notebook for model development
│   ├── model_development.py       # Python script for model training
│   ├── titanic_survival_model.pkl # Trained Random Forest model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder_sex.pkl      # Sex label encoder
│   └── label_encoder_embarked.pkl # Embarked label encoder
│
├── static/
│   └── style.css                  # CSS styling
│
└── templates/
    └── index.html                 # HTML template
```

## 🤖 Model Information

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: ~85%
- **Testing Accuracy**: ~82%
- **Persistence Method**: Joblib
- **Features**: 5 selected features from Titanic dataset

## 🎨 Web Interface

The web application features:
- Modern, responsive design
- Interactive form for passenger details
- Real-time prediction results
- Animated probability visualization
- Mobile-friendly interface

## 📊 Model Development

The model development process includes:
1. Data loading and exploration
2. Missing value imputation
3. Feature selection and encoding
4. Feature scaling with StandardScaler
5. Train-test split (80/20)
6. Random Forest training with optimized hyperparameters
7. Model evaluation with classification report
8. Model persistence with Joblib

## 🌐 Deployment

### Deployment Options

This application can be deployed on:
- **Render.com** (Recommended)
- **PythonAnywhere.com**
- **Streamlit Cloud**
- **Vercel**
- **Scorac.com**

### Render.com Deployment Steps

1. Create a `render.yaml` file (already included)
2. Push code to GitHub
3. Connect GitHub repository to Render
4. Deploy automatically

### Environment Variables

No environment variables required for basic deployment.

## 📝 Usage

1. Open the web application
2. Fill in passenger details:
   - Select passenger class (1st, 2nd, or 3rd)
   - Choose gender
   - Enter age
   - Enter ticket fare
   - Select port of embarkation
3. Click "Predict Survival"
4. View the prediction result with probability scores

## 🧪 Testing

To test the model manually:

```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Example: 1st class female, age 29, fare 100, embarked C
features = np.array([[1, 0, 29, 100, 0]])
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)
print("Survived" if prediction[0] == 1 else "Did Not Survive")
```

## 📈 Model Performance

The Random Forest Classifier achieves:
- High precision for both survival classes
- Balanced recall across classes
- F1-scores above 0.80 for both classes
- Robust performance on unseen data

## 🛠️ Technologies Used

- **Backend**: Flask 3.0.0
- **ML Libraries**: scikit-learn, pandas, numpy
- **Model Persistence**: Joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Matplotlib, Seaborn (for notebook)

## 👤 Author

**[Your Name]**
- Matric Number: [Your Matric Number]
- GitHub: [Your GitHub Profile]

## 📄 License

This project is created for educational purposes as part of a machine learning course assignment.

## 🙏 Acknowledgments

- Dataset: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Kaggle community for dataset maintenance
- Course instructors for project guidance
