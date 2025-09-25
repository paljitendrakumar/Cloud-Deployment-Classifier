# Cloud-Deployment-Classifier
This classifier recommends the optimal cloud deployment (Container vs. VM) based on workload specs. It uses a Random Forest model to provide instant predictions and features an interactive dashboard with performance insights. Users can also upload their own custom dataset for analysis.
☁️ Cloud Deployment AI Classifier
This Streamlit application uses a Random Forest model to recommend whether a given workload is better suited for a Container or a Virtual Machine (VM) deployment.

✨ Features
AI-Powered Recommendations: Get instant deployment suggestions based on workload characteristics.
Interactive Dashboard: Visualize model performance, feature importance, and data distributions.
Custom Dataset Upload: Upload your own CSV file to train and evaluate the model on your data.
Data Exploration: Filter and view the training data directly in the app.
🚀 How to Run Locally
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
Install the required libraries:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run app.py
