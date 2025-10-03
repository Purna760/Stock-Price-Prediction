import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("🤖 Mobile Machine Learning App")
st.write("Build and deploy ML models directly from your phone!")

# Sample dataset or file upload
st.sidebar.header("1. Data Input")
data_option = st.sidebar.radio("Choose data source:", 
                              ["Use Sample Data", "Upload Your Data"])

if data_option == "Use Sample Data":
    # Load sample dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    target_names = iris.target_names
    
    st.subheader("Sample Iris Dataset")
    st.write("**Features:**", list(X.columns))
    st.write("**Target classes:**", list(target_names))
    st.dataframe(pd.concat([X, y], axis=1).head())
    
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(data.head())
        
        # Let user select features and target
        features = st.multiselect("Select features", data.columns)
        target = st.selectbox("Select target variable", data.columns)
        
        if features and target:
            X = data[features]
            y = data[target]
            target_names = y.unique()
    else:
        st.info("Please upload a CSV file or use sample data")
        st.stop()

# Model configuration
st.sidebar.header("2. Model Configuration")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

if st.sidebar.button("🚀 Train Model"):
    if 'X' in locals() and 'y' in locals():
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            st.success(f"✅ Model trained successfully!")
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            st.subheader("📊 Feature Importance")
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots()
            ax.barh(feature_imp['feature'], feature_imp['importance'])
            ax.set_xlabel('Importance')
            st.pyplot(fig)
            
            # Prediction interface
            st.subheader("🔮 Make Predictions")
            st.write("Enter feature values for prediction:")
            
            col1, col2 = st.columns(2)
            input_features = {}
            
            for i, feature in enumerate(X.columns):
                with col1 if i % 2 == 0 else col2:
                    input_features[feature] = st.number_input(
                        f"{feature}",
                        value=float(X[feature].mean()),
                        step=0.1
                    )
            
            if st.button("Predict"):
                input_array = np.array([list(input_features.values())])
                prediction = model.predict(input_array)[0]
                proba = model.predict_proba(input_array)[0]
                
                if 'target_names' in locals():
                    predicted_class = target_names[prediction]
                    st.success(f"Predicted class: **{predicted_class}**")
                else:
                    st.success(f"Predicted value: **{prediction}**")
                
                # Show probabilities
                st.write("Prediction probabilities:")
                for i, prob in enumerate(proba):
                    class_name = target_names[i] if 'target_names' in locals() else f"Class {i}"
                    st.write(f"{class_name}: {prob:.2%}")

else:
    st.info("👈 Configure your model in the sidebar and click 'Train Model'")

# Instructions
st.sidebar.header("ℹ️ Instructions")
st.sidebar.info("""
1. Choose data source
2. Configure model parameters
3. Click 'Train Model'
4. View results & make predictions
""")
