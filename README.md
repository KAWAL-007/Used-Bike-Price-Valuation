🏍️ Used Bike Valuation & Analytics Dashboard
An interactive, end-to-end Machine Learning web application built with Streamlit, Scikit-Learn, and Plotly. This application automates data preprocessing, trains a regression model, estimates fair market values for pre-owned motorcycles, and provides interactive exploratory data analysis (EDA).

📌 Features
🤖 Automated Machine Learning Pipeline: Preprocesses data, builds a RandomForestRegressor pipeline, and serializes the model automatically on first run.

💰 Interactive Price Valuation Tool: Predicts bike prices based on location, ownership history, age, engine capacity (cc), and mileage.

📊 Market EDA Analytics: Visualizes market distributions, brand popularity, price vs. engine power, and value depreciation over time using interactive Plotly charts.

⚡ Single-File Architecture: Self-contained Streamlit application that handles both back-end model preparation and front-end dashboard rendering.

🛠️ Tech Stack
Language: Python

Frontend/Dashboard: Streamlit

Machine Learning: Scikit-Learn

Data Processing: Pandas, NumPy

Data Visualization: Plotly

📂 Project Structure
Plaintext
├── app.py                 # Main Streamlit application file
├── Used_Bikes.csv         # Dataset (required for training and EDA)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
🚀 Getting Started
1. Clone the Repository
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required libraries using:

Bash
pip install -r requirements.txt
3. Add Dataset
Ensure Used_Bikes.csv is placed in the root directory alongside app.py.

4. Run the Application
Launch the Streamlit app:

Bash
streamlit run app.py
🖥️ Application Preview
Price Valuation: Input vehicle details like brand, engine CC, age, and kilometers driven to receive an instant market valuation.

EDA Analytics: Explore dataset trends through dynamic graphs and insights.
