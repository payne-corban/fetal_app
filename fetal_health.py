# Import necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the model and dataset with caching
@st.cache_resource
def load_model(model_name):
    with open(f'{model_name}.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

fetal_df= pd.read_csv('fetal_health.csv')

# Streamlit app
def main():
    st.title("Fetal Health Classification: A Machine Learning App")

    # Display an image of fetus
    st.image('fetal_health_image.gif', width=400)
    
    st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

    
    # Create a sidebar for input collection
    st.sidebar.header('Fetal Health Features Input')
    st.sidebar.write('Upload your data')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.write("Example of the file format to upload:")
    st.write(fetal_df.head())


    # Select the ML model
    ml_selection = st.sidebar.selectbox('ML Algorithm Type', options=['Decision Tree', 'Random Forest','AdaBoost','Soft Voting'])


    # Load the selected model based on user input
    if ml_selection == 'Decision Tree':
        clf = load_model('dt_fetal')
    elif ml_selection == 'Random Forest':
        clf = load_model('rf_fetal')
    elif ml_selection == 'AdaBoost':
        clf = load_model('ada_fetal')
    elif ml_selection == 'Soft Voting':
        clf = load_model('soft_fetal')
    else:
        raise ValueError("Invalid model selection. Choose from 'Decision Tree', 'Random Forest', 'AdaBoost', or 'SVM'")

    











    




   



    # Showing additional items in tabs for corresponding ML model
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])



# Tab 1: Feature Importance Visualization

    # Choose the appropriate feature importance image based on model selection
    if ml_selection == 'Decision Tree':
        st.image('feature_imp_dt.svg')
    elif ml_selection == 'Random Forest':
        st.image('feature_imp_rf.svg')
    elif ml_selection == 'AdaBoost':
        st.image('feature_imp_ada.svg')
    elif ml_selection == 'SVM':
        st.write("Feature importance is not available for SVM.")
    else:
        st.write("Invalid model selection.")

    st.caption("Features used in this prediction are ranked by relative importance.")







 # Tab 2: Confusion Matrix

    # Choose the appropriate confusion matrix image based on model selection
    if ml_selection == 'Decision Tree':
        st.image('confusion_mat_fetal_dt.svg')
    elif ml_selection == 'Random Forest':
        st.image('confusion_mat_fetal_rf.svg')
    elif ml_selection == 'AdaBoost':
        st.image('confusion_mat_fetal_ada.svg')
    elif ml_selection == 'SVM':
        st.image('confusion_mat_fetal_svm.svg')
    else:
        st.write("Invalid model selection.")

    st.caption("Confusion Matrix of model predictions.")





# Tab 3: Classification Report

    # Choose the appropriate classification report file based on model selection
    if ml_selection == 'Decision Tree':
        report_file = 'fetal_class_report_dt.csv'
    elif ml_selection == 'Random Forest':
        report_file = 'fetal_class_report_rf.csv'
    elif ml_selection == 'AdaBoost':
        report_file = 'fetal_class_report_ada.csv'
    elif ml_selection == 'Soft Voting':
        report_file = 'fetal_class_report_svm.csv'
    else:
        st.write("Invalid model selection.")
        report_file = None

    # Display classification report if a valid file was selected
    if report_file:
        report_df = pd.read_csv(report_file, index_col=0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each category.")


if __name__ == "__main__":
    main()
