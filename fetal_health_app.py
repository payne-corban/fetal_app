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

fetal_df = pd.read_csv('fetal_health.csv')

# Streamlit app
def main():
    st.title("Fetal Health Classification: A Machine Learning App")

    # Display an image of fetus
    st.image('fetal_health_image.gif', width=800)
    
    st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")
    
    # Create a sidebar for input collection
    st.sidebar.header('Fetal Health Features Input')
    uploaded_file = st.sidebar.file_uploader("Upload your data", type=["csv"], help='File must be in CSV format')
    st.sidebar.warning('Ensure your data strictly  follows the format outlined below')
    st.sidebar.write("Example of the file format to upload:")
    st.sidebar.write(fetal_df.head())

    # Radio button to select the ML model
    ml_selection = st.sidebar.radio('Choose Model for Prediction', options=['Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'])
    
    # Displaying selected model
    if ml_selection == 'Decision Tree':
        clf = load_model('dt_fetal'), st.sidebar.info('You selected: Decision Tree')
    elif ml_selection == 'Random Forest':
        st.sidebar.info('You selected: Random Forest')
    elif ml_selection == 'AdaBoost':
        st.sidebar.info('You selected: AdaBoost')
    elif ml_selection == 'Soft Voting':
        st.sidebar.info('You selected: Soft Voting')
    else:
        raise ValueError("Invalid model selection. Choose from 'Decision Tree', 'Random Forest', 'AdaBoost', or 'Soft Voting'.")

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
        raise ValueError("Invalid model selection. Choose from 'Decision Tree', 'Random Forest', 'AdaBoost', or 'Soft Voting'.")

    # Process the uploaded file if available
    if uploaded_file:
        # Load the uploaded file
        input_data = pd.read_csv(uploaded_file)
        
        # Make predictions
        predictions = clf.predict(input_data)
        
        # Get prediction probabilities
        prediction_probs = clf.predict_proba(input_data)
        
        # Map the prediction to class labels
        label_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        input_data['Predicted Fetal Health'] = [label_mapping[pred] for pred in predictions]
        
        # Add prediction probability column
        input_data['Prediction Probability (%)'] = np.max(prediction_probs, axis=1) * 100
        
        # Used Chatgpt to help apply the background color based on prediction
        def color_predicted_class(val):
            if val == "Normal":
                return 'background-color: lime'
            elif val == "Suspect":
                return 'background-color: yellow'
            elif val == "Pathological":
                return 'background-color: orange'
            else:
                return ''

        # Apply the color to the column
        color_df = input_data.style.applymap(color_predicted_class, subset=['Predicted Fetal Health'])

        # Display updated data with predictions and apply the styling
        st.write("### Uploaded Data with Predictions")
        st.dataframe(color_df)


        # Showing additional items in tabs for corresponding ML model
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            if ml_selection == 'Decision Tree':
                st.image('feature_imp_dt.svg')
            elif ml_selection == 'Random Forest':
                st.image('feature_imp_rf.svg')
            elif ml_selection == 'AdaBoost':
                st.image('feature_imp_ada.svg')
            elif ml_selection == 'Soft Voting':
                st.image('feature_imp_voting.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            if ml_selection == 'Decision Tree':
                st.image('confusion_mat_fetal_dt.svg')
            elif ml_selection == 'Random Forest':
                st.image('confusion_mat_fetal_rf.svg')
            elif ml_selection == 'AdaBoost':
                st.image('confusion_mat_fetal_ada.svg')
            elif ml_selection == 'Soft Voting':
                st.image('confusion_mat_fetal_svc.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            if ml_selection == 'Decision Tree':
                report_file = 'fetal_class_report_dt.csv'
            elif ml_selection == 'Random Forest':
                report_file = 'fetal_class_report_rf.csv'
            elif ml_selection == 'AdaBoost':
                report_file = 'fetal_class_report_ada.csv'
            elif ml_selection == 'Soft Voting':
                report_file = 'fetal_class_report_svc.csv'
            
            if report_file:
                report_df = pd.read_csv(report_file, index_col=0).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
                st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each category.")

    else:
        # Show info icon with message
        st.info("Please upload data to proceed.")

if __name__ == "__main__":
    main()
