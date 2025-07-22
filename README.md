Employee Salary Prediction: Ensuring Fair and Accurate Compensation
Organizations across various sectors face the complex challenge of determining fair and accurate employee salaries. This project offers a data-driven solution, leveraging supervised classification models to predict salary trends based on key factors like age, education, occupation, working hours, and years of experience. By analyzing these attributes, the system aims to promote transparency and equity in compensation, addressing issues of dissatisfaction, imbalance, and inefficiencies caused by inconsistent salary structures.


Presented By:
Manmaya Prasad Hotta 
AICTE Internship Student Regd ID: STU665ae9d3dea901717234131
Department of Computer Science and Engineering
Trident Academy of Technology

Key Features
Intelligent Salary Forecasting: Utilizes machine learning models to predict employee salaries.
Robust Data Preprocessing: Comprehensive handling of missing values, removal of irrelevant data, and outlier detection ensure high-quality training data.
Comparative Model Analysis: Explores and evaluates five popular machine learning algorithms: Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Gradient Boosting Classifier.
Intuitive Web Application: A user-friendly Streamlit interface enables real-time single employee predictions and supports batch predictions via CSV file uploads.
Scalable & Accessible: Designed for local deployment and prepared for cloud integration, ensuring accessibility for both technical and non-technical users.

Technologies Used
Python 3.8+ 
Pandas: For efficient data manipulation and analysis.
Joblib: For saving and loading trained models.
Scikit-learn: The core library for building and training machine learning models.
Streamlit: For creating the interactive web application.
NumPy: For numerical operations.
Matplotlib: For data visualization, particularly outlier detection.

Getting Started
System Requirements

Operating System: Windows 10 / Linux / macOS 
Processor: Intel Core i3 or higher 
RAM: Minimum 4 GB (8 GB recommended) 
Storage: At least 500 MB of free space for libraries, data, and model files 

Installation
Clone the repository:
Bash
git clone https://github.com/manmayahotta/IBM-SkillsBuild-Employee-Salary-Prediction.git
cd IBM-SkillsBuild-Employee-Salary-Prediction
Create a virtual environment (recommended):

Bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
Install the required libraries:

Bash
pip install pandas joblib scikit-learn streamlit numpy matplotlib
Usage
Run the Streamlit application:

Bash
streamlit run your_app_file_name.py # Replace 'your_app_file_name.py' with the actual name of your Streamlit app file
The application will automatically open in your default web browser. You can then:
Predict Single Employee Salary: Input individual employee details into the sidebar fields (e.g., age, education, occupation, hours per week, experience) to receive an instant salary class prediction.
Perform Batch Prediction: Upload a .csv file containing multiple employee records. The system will process the file, predict salary classes for all entries, and allow you to download the results as a new CSV file.

Project Workflow
The project follows a systematic machine learning workflow:
Data Collection: Loading the adult 3.csv dataset.
Data Preprocessing: Handling missing values by replacing '?' with "Others" , removing irrelevant data like 'Without-pay' and 'Never-worked' workclass values , and performing outlier detection and removal in 
age, educational-num, capital-gain, and hours-per-week.
Feature Engineering: Selecting and transforming features, including dropping the education column and applying Label Encoding to categorical variables.
Model Training: Splitting data into 80% training and 20% testing sets , selecting five popular machine learning models , creating pipelines with 
StandardScaler , and training all models.
Performance Evaluation: Assessing models using accuracy_score() and classification_report().
Model Deployment: Selecting and saving the best model , then deploying via a Streamlit web application with both single and batch prediction capabilities.

Conclusion
This project successfully demonstrates a complete machine learning workflow, from data preprocessing to model deployment, to effectively predict employee salaries. By ensuring data quality through cleaning, encoding, and outlier handling , and integrating a user-friendly Streamlit interface for real-time and batch predictions , this system highlights the critical role of data-driven approaches in modern HR and organizational decision-making. It serves as a strong foundation for future advancements in salary forecasting.

Future Scope
The project has significant potential for future enhancements:
Real-Time Data Integration: Connecting with live HR management tools for continuous model learning and adaptation.
Advanced Model Enhancement: Exploring sophisticated machine learning techniques like Random Forest, Gradient Boosting, and Deep Neural Networks, and optimizing performance with Grid Search and Cross-Validation.
Expanded Feature Set: Incorporating additional predictive features such as job location, industry sector, company size, performance scores, and market benchmarks.
Cross-Platform & Multilingual Accessibility: Extending the Streamlit application to mobile platforms (Android/iOS) and supporting multiple languages.
Cloud Deployment & API Access: Deploying on scalable cloud platforms (AWS, Azure, Google Cloud) and providing secure RESTful APIs for broader system integration.

Repository
For complete code, datasets, and detailed documentation, please visit the GitHub repository:
https://github.com/manmayahotta/IBM-SkillsBuild-Employee-Salary-Prediction.git 
