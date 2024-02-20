import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_users = 100
num_months = 3
user_ids = np.repeat(np.arange(1, num_users + 1), num_months)
months = np.tile(np.arange(1, num_months + 1), num_users)

# Generate Weight and Height data
weight_kg = np.random.uniform(50, 120, len(user_ids))
height_cm = np.random.uniform(150, 200, len(user_ids))

# Calculate BMI
bmi = weight_kg / ((height_cm / 100) ** 2)

# Create DataFrame
users_data = pd.DataFrame({
    'User_ID': user_ids,
    'Month': months,
    'Weight_kg': weight_kg,
    'Height_cm': height_cm,
    'BMI': bmi
})


# Sidebar with options
st.sidebar.header('Analysis Options')
analysis_option = st.sidebar.selectbox('Select Analysis:', ('BMI Distribution',
                                                           'Correlation Analysis',
                                                           'Comparison with Norms',
                                                           'User Profiles and Recommendations','BMI Improvement'))

# Main content based on selected option
st.title('Campus Health and Wellness Analytics')

if analysis_option == 'BMI Distribution':
    st.header('BMI Distribution Analysis')
    users_data['BMI'] = users_data['Weight_kg'] / ((users_data['Height_cm'] / 100) ** 2)

    # Visualize BMI distribution
    st.write('BMI Distribution')
    fig, ax = plt.subplots()
    sns.histplot(users_data['BMI'], kde=True, bins=20, ax=ax)
    st.pyplot(fig)


elif analysis_option == 'Correlation Analysis':
    st.header('Correlation Between Weight/Height and Other Metrics')

    # Generate dummy data for other health-related metrics
    num_rows = len(users_data)
    users_data['Fitness_Level'] = np.random.randint(1, 11, num_rows)  # Dummy fitness level data (1-10 scale)
    users_data['Mental_Health_Score'] = np.random.randint(1, 11,
                                                          num_rows)  # Dummy mental health score data (1-10 scale)
    users_data['Diet_Habits'] = np.random.choice(['Healthy', 'Moderate', 'Unhealthy'],
                                                 num_rows)  # Dummy diet habits data

    # Calculate correlation matrix
    correlation_matrix = users_data[['Weight_kg', 'Height_cm', 'Fitness_Level', 'Mental_Health_Score']].corr()

    # Visualize correlation matrix
    st.write('Correlation Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


elif analysis_option == 'Comparison with Norms':
    st.header('Comparison with Population Norms')

    # Ensure 'BMI' column exists
    if 'BMI' not in users_data:
        st.write('Please select BMI Distribution analysis first to generate BMI data.')
    else:
        # Generate dummy data for comparison with norms
        users_data['BMI_Category'] = pd.cut(users_data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

        # Visualize comparison with norms
        st.write('Comparison with BMI Norms')
        fig, ax = plt.subplots()
        sns.countplot(x='BMI_Category', data=users_data, order=['Underweight', 'Normal', 'Overweight', 'Obese'])
        st.pyplot(fig)


elif analysis_option == 'User Profiles and Recommendations':
    st.header('User Profiles and Health Recommendations')

    # Generate dummy data for user profiles and recommendations
    num_rows = len(users_data)
    users_data['Health_Recommendations'] = np.random.choice(['Increase Physical Activity', 'Improve Diet Habits',
                                                             'Engage in Mindfulness Activities'], num_rows)

    # Visualize user profiles and recommendations
    st.write('User Health Recommendations')
    st.table(users_data[['User_ID', 'Weight_kg', 'Height_cm', 'Health_Recommendations']].sample(10))

elif analysis_option == 'BMI Improvement':
    st.header('Individual BMI Improvement Month on Month')

    # Visualize individual BMI improvement month on month
    st.write('Individual BMI Improvement Month on Month')

    # Dropdown to select user
    selected_user = st.sidebar.selectbox('Select User:', users_data['User_ID'].unique())

    # Get data for selected user
    selected_user_data = users_data[users_data['User_ID'] == selected_user]

    # Button to show graph
    if st.sidebar.button('Show BMI Improvement Graph'):
        # Group by user and month to calculate BMI for each user for each month
        monthly_bmi = selected_user_data.groupby(['User_ID', 'Month'])['BMI'].mean().reset_index()

        # Plot BMI improvement month on month for selected user
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_bmi.plot(x='Month', y='BMI', marker='o', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('BMI')
        ax.set_title(f'BMI Improvement for User {selected_user} Month on Month')
        st.pyplot(fig)
