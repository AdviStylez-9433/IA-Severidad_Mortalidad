# Clinical Severity and Mortality Risk Calculator

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)
![Machine Learning](https://img.shields.io/badge/machine-learning-orange)

A web-based tool for calculating patient severity and mortality risk based on vital signs, medical history, and present symptoms.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Clinical Parameters](#clinical-parameters)
- [Technology Stack](#technology-stack)
- [Screenshots](#screenshots)
- [Limitations](#limitations)
- [License](#license)

## Features

✅ Real-time risk calculation  
✅ Interactive visualization  
✅ Responsive design  
✅ Clinical interpretation  
✅ Model-based scoring  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/clinical-risk-calculator.git
   cd clinical-risk-calculator
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the development server:
   ```bash
   python app.py
   python -m http.server 8000
   ```

Access the application at:
   ```
   http://localhost:5000
   ```

## API Endpoints

| Endpoint     | Method | Description                        |
|-------------|--------|------------------------------------|
| `/`         | GET    | Main application interface        |
| `/calculate` | POST   | Calculate risk scores (JSON)      |

## Clinical Parameters

### Vital Signs
- Heart rate (bpm)
- Blood pressure (mmHg)
- Respiratory rate (rpm)
- Temperature (°C)
- Oxygen saturation (%)

### Medical History
- Diabetes
- Hypertension
- Heart disease
- Age

## Technology Stack

### Backend
- Python 3
- Flask
- Scikit-learn
- NumPy

### Frontend
- HTML5
- CSS3
- JavaScript (ES6)

## Screenshots

### Form Screenshot
_Input form for patient data_

### Results Screenshot
_Risk visualization and interpretation_

## Limitations

⚠️ **Important:** This is a demonstration project only. Not for clinical use.

- Uses synthetic training data
- Limited clinical parameters
- Not validated for medical use
- Basic scoring algorithm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Additional Recommendations

1. Create a `screenshots` folder and add actual screenshots of your application.
2. Add a `requirements.txt` file with:
   ```
   flask==2.0.1
   scikit-learn==0.24.2
   numpy==1.21.2
   ```
3. For better presentation, consider:
   - Adding actual screenshots
   - Including a demo GIF/video
   - Adding contribution guidelines
   - Including testing instructions
   - Adding deployment instructions

_Disclaimer: This tool is for educational purposes only and should not be used for actual medical decision-making. Always consult with qualified healthcare professionals for medical assessments._
