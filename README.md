
Employee Salary Prediction based on Supervised Classification Models
This project implements an employee salary prediction system using supervised classification models, addressing the complex challenge organizations face in determining fair and accurate employee salaries. The system analyzes various factors such as age, education, occupation, working hours, and years of experience to predict salary trends. An inconsistent salary structure can lead to dissatisfaction and inefficiencies in workforce management, highlighting the need for data analysis to understand patterns and predict salary trends effectively. Identifying key attributes influencing earnings is crucial for transparency and equity in compensation.





The project demonstrates a complete machine learning workflow, from data preprocessing to model deployment, and utilizes a user-friendly Streamlit interface for real-time and batch salary predictions. This data-driven approach aims to assist HR and organizational decision-making.



Table of Contents
Problem Statement

System Development Approach

System Requirements

Libraries Required

Algorithm & Deployment

Data Collection

Handling Missing Values

Irrelevant Data Removal

Outlier Detection & Removal

Feature Selection & Transformation

Data Splitting

Model Selection

Pipeline Creation

Model Training & Prediction

Performance Evaluation

Best Model Selection

Web Application Using Streamlit

Batch Prediction Module

Deployment & Accessibility

Results

Project Workflow

Conclusion

Future Scope

References

Problem Statement
Determining fair and accurate employee salaries is a complex challenge for organizations. Salaries are influenced by multiple factors, including age, education, occupation, working hours, and years of experience. Inconsistent salary structures can lead to dissatisfaction, imbalance, and inefficiencies in workforce management. There is a growing need to analyze employee-related data to understand patterns and predict salary trends more effectively. Identifying the key attributes that influence employee earnings is essential for ensuring transparency and equity in compensation.





System Development Approach
System Requirements

Operating System: Windows 10 / Linux / macOS 


Processor: Intel Core i3 or higher 


RAM: Minimum 4 GB (8 GB recommended) 


Software: Python 3.8 or above, Streamlit, and a web browser 


Storage: At least 500 MB of free space for libraries, data, and model files 

Libraries Required

Pandas: For data manipulation and analysis 


Joblib: For saving and loading the trained model 


Scikit-learn: For building and training machine learning models 


Streamlit: For creating the interactive web application 


NumPy: For numerical operations (if used during preprocessing/modeling) 

Algorithm & Deployment
The system development involved several key steps:

Data Collection
The 

adult 3.csv dataset, a variant of the UCI Adult Income Dataset, was used. It was loaded using 

pandas.read_csv() for analysis and preprocessing. Initial insights were gathered using 

data.head(), data.tail(), and data.shape to preview entries and understand dataset dimensions.

Handling Missing Values
Missing values, represented by '?' in the 

workclass and occupation columns, were identified and replaced with "Others" using the replace() method to maintain data integrity. The absence of NaN values was verified using 

data.isna().sum() for all columns.

Irrelevant Data Removal
Rows where 

workclass values were 'Without-pay' or 'Never-worked' were removed to ensure the relevance of workforce data for salary prediction.

Outlier Detection & Removal
Outliers in 'age', 'educational-num', 'capital-gain', and 'hours-per-week' were visualized using 

matplotlib.pyplot.boxplot(). Filtering conditions were applied to remove extreme values: 'age' was filtered between 17 and 75, and 'educational-num' was restricted between 5 and 16, ensuring data consistency and reduced noise for better model performance.


Feature Selection & Transformation
The 'education' column was dropped due to its high correlation with 'educational-num'. Label Encoding was applied to categorical variables such as 

workclass, marital-status, occupation, relationship, race, gender, and native-country, enabling machine learning models to process this data.

Data Splitting
The dataset was divided into an 80% training set and a 20% testing set. 

train_test_split() from sklearn.model_selection was used with a fixed random_state for reproducibility.

Model Selection
Five popular machine learning models were chosen for evaluation: 

Logistic Regression 

Random Forest 

K-Nearest Neighbors (KNN) 

Support Vector Machine (SVM) 

Gradient Boosting Classifier 

Pipeline Creation
A 

Pipeline from sklearn.pipeline was created for each model, including StandardScaler for feature scaling, combining preprocessing and the model into a single pipeline.

Model Training & Prediction
All selected models were trained using the training data (

X_train, y_train). 

pipeline.fit() was used for training, and pipeline.predict() for predictions.

Performance Evaluation
Model performance was evaluated using 

accuracy_score() to measure correct predictions and classification_report() to analyze precision, recall, and F1-score.

Best Model Selection
Models were compared, and the one with the highest accuracy on the test set and balanced precision and recall was selected. The best model was saved using 

joblib.dump() for deployment in the Streamlit application.

Web Application Using Streamlit
A web application was developed using the Streamlit framework. It features sidebar inputs for 

age, education, occupation, hours/week, and experience, providing real-time predictions using the loaded model and displaying results in a success box.

Batch Prediction Module
A batch prediction module allows users to upload a 

.csv file with employee data, predict salary classes for all rows, and display the output table. Users can also download the prediction results as a CSV file.


Deployment & Accessibility
The Streamlit application was deployed for local use and prepared for cloud deployment, providing a smooth, user-friendly interface for both technical and non-technical users to make predictions.

Results
The models were evaluated, and the Gradient Boosting Classifier was identified as the best model with an accuracy of 0.8557.

(The PPT included screenshots of code and results, including data previews, missing value checks, outlier detection plots, feature transformation, and model comparison. These visual results are implied here, as they cannot be directly rendered in text.)

Project Workflow
The project workflow involves steps from data ingestion and cleaning to model training, evaluation, and deployment through a Streamlit application.


(The PPT included screenshots of the Streamlit application, showing single employee prediction and batch prediction functionalities. These visual representations of the workflow are implied here.)

Conclusion
This project successfully demonstrates a complete machine learning workflow, from data preprocessing to model deployment. Various regression algorithms were explored and evaluated for effective employee salary prediction. Data cleaning, encoding, and outlier handling ensured the model was trained on high-quality data. The integration of a user-friendly Streamlit interface enables both real-time and batch salary predictions. This system highlights the role of data-driven approaches in assisting HR and organizational decision-making and can serve as a foundation for future enhancements in salary forecasting systems.





Future Scope

Real-Time Data Integration: Future enhancements could include integrating the system with real-time HR management tools for continuous learning and adaptation from live employee data, improving accuracy and relevance over time.


Advanced Model Enhancement: Beyond the current models, future iterations can leverage advanced machine learning techniques like Random Forest, Gradient Boosting, and Deep Neural Networks, along with techniques like Grid Search and Cross-Validation for optimized model performance.


Expanded Feature Set: Incorporating more predictive features such as job location, industry sector, company size, work performance scores, and current market salary benchmarks can make the model more robust and context-aware.


Cross-Platform & Multilingual Accessibility: The Streamlit application can be extended to mobile platforms (Android/iOS) and support multiple languages to increase accessibility for HR teams and users across different regions.

Cloud Deployment & API Access: Future deployment on cloud platforms like AWS, Azure, or Google Cloud can provide scalable infrastructure. Secure RESTful APIs can allow other systems or web applications to access the model as a service, ensuring scalability and wider integration.

References

Scikit-Learn: Machine learning library used for implementing models and evaluating metrics. 


Pandas: Used for data loading, preprocessing, and manipulation. 


NumPy: Provided numerical computation support. 


Matplotlib & Seaborn: Utilized for exploratory data analysis and plotting. 


Streamlit: Used to build and deploy the interactive web application. 

For complete code, datasets, and documentation, please visit the GitHub repository: 

https://github.com/manmayahotta/IBM-SkillsBuild-Employee-Salary-Prediction.git 
