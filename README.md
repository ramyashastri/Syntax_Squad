## Syntax Squad: Career Prediction Model
This project was developed by the team Syntax Squad during an 8-hour hackathon, aimed at helping students and young professionals discover their ideal career paths using machine learning.

## Key Features:
* Multi-factor analysis with 10+ input parameters
* 91%+ accuracy based on Chi-Square Test Analysis
* Probability insights for multiple career options
* Interactive web interface with Gradio
* Modern, responsive UI
  
## Tech Stack Organized:
* Core Technologies: Python, XGBoost, scikit-learn, Gradio
* Data Processing: Pandas, NumPy, Matplotlib/Seaborn
* Additional Libraries: LightGBM, CatBoost

## Project Overview

Syntax Squad leverages advanced machine learning algorithms to predict the most suitable career for users based on their academic background, skills, preferences, and personal attributes.  
The model is trained on a diverse dataset and supports interactive predictions via a user-friendly web interface.

## Live Demo

Try out the model here:  
[https://huggingface.co/spaces/WickedFaith/Syntax-Squad](https://huggingface.co/spaces/WickedFaith/Syntax-Squad)

## How to Run Locally

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/syntax-squad.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the app:
    ```bash
    python app.py
    ```

## Project Structure

- `career_prediction_model.ipynb` – Model training and evaluation
- `deployment/app.py` – Gradio web app for predictions
- `final_train.csv` – Training dataset
- `deployment/` – Deployment scripts and saved models

