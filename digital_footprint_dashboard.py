import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st

# Load the dataset
df = pd.read_csv('digital_footprint_sample_dataset.csv')

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Device', 'Location', 'Footprint_Type'], drop_first=True)

# Features and target
#X = df_encoded.drop(['User_ID', 'Risk_Score'], axis=1)
X = df_encoded.drop(['User_ID', 'IP_Address', 'Risk_Score'], axis=1)
y = df_encoded['Risk_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict risk scores
df['Predicted_Risk_Score'] = model.predict(X)

# Calculate error on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Risk Profiling Logic
def risk_category(score):
    if score >= 0.7:
        return 'High Risk'
    elif score >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

# Apply risk profiling
df['Risk_Category'] = df['Predicted_Risk_Score'].apply(risk_category)

# Alerts Setup
alerts = df[df['Risk_Category'] == 'High Risk']

# Streamlit App
st.title('Digital Footprint Risk Analysis Dashboard')

st.write(f'Model Mean Squared Error: {mse:.4f}')

st.sidebar.header('Filter Options')
selected_device = st.sidebar.multiselect('Select Device', options=df['Device'].unique(), default=df['Device'].unique())
selected_location = st.sidebar.multiselect('Select Location', options=df['Location'].unique(), default=df['Location'].unique())
selected_risk = st.sidebar.multiselect('Select Risk Category', options=['High Risk', 'Medium Risk', 'Low Risk'], default=['High Risk', 'Medium Risk', 'Low Risk'])

# Filter data
filtered_df = df[(df['Device'].isin(selected_device)) & (df['Location'].isin(selected_location)) & (df['Risk_Category'].isin(selected_risk))]

# Display filtered data
st.write('Filtered User Data', filtered_df)

# Plot 1: Risk Category Distribution
st.subheader('User Risk Category Distribution')
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x='Risk_Category', palette='coolwarm', order=['High Risk', 'Medium Risk', 'Low Risk'], ax=ax1)
ax1.set_xlabel('Risk Category')
ax1.set_ylabel('Number of Users')
st.pyplot(fig1)

# Plot 2: Device Usage Distribution
st.subheader('Device Usage Distribution')
fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x='Device', palette='viridis', ax=ax2)
ax2.set_xlabel('Device Type')
ax2.set_ylabel('Number of Users')
st.pyplot(fig2)

# Plot 3: Average Predicted Risk Score per Location
st.subheader('Average Predicted Risk Score per Location')
fig3, ax3 = plt.subplots()
sns.barplot(data=filtered_df, x='Location', y='Predicted_Risk_Score', palette='magma', ci=None, ax=ax3)
ax3.set_xlabel('Location')
ax3.set_ylabel('Average Predicted Risk Score')
st.pyplot(fig3)

# Plot 4: Login Times vs Purchases by Risk Category
st.subheader('Login Times vs Purchases by Risk Category')
fig4, ax4 = plt.subplots()
sns.scatterplot(data=filtered_df, x='Login_Times', y='Purchases', hue='Risk_Category', palette='Set1', ax=ax4)
ax4.set_xlabel('Login Times')
ax4.set_ylabel('Purchases')
st.pyplot(fig4)

# Display high risk alerts
st.subheader('High Risk User Alerts')
st.write(alerts[['User_ID', 'IP_Address', 'Predicted_Risk_Score', 'Risk_Category']])

# Download high risk alerts
csv = alerts.to_csv(index=False)
st.download_button(
    label="Download High Risk Alerts as CSV",
    data=csv,
    file_name='high_risk_user_alerts.csv',
    mime='text/csv'
)
