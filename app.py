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
analysis_option = st.sidebar.selectbox('Select Analysis:', ('Average Weight and Height', 'BMI Distribution',
                                                           'Weight and Height Trends', 'Correlation Analysis',
                                                           'Goals Progress Tracking', 'Comparison with Norms',
                                                           'User Profiles and Recommendations','BMI Improvement'))

# Main content based on selected option
st.title('Campus Health and Wellness Analytics')

if analysis_option == 'Average Weight and Height':
    st.header('Average Weight and Height Analysis')
    avg_weight = users_data['Weight_kg'].mean()
    avg_height = users_data['Height_cm'].mean()
    st.write(f"Average Weight: {avg_weight:.2f} kg")
    st.write(f"Average Height: {avg_height:.2f} cm")

    # Visualize average weight and height
    fig, ax = plt.subplots()
    ax.bar(['Average Weight', 'Average Height'], [avg_weight, avg_height])
    st.pyplot(fig)


elif analysis_option == 'BMI Distribution':
    st.header('BMI Distribution Analysis')
    users_data['BMI'] = users_data['Weight_kg'] / ((users_data['Height_cm'] / 100) ** 2)

    # Visualize BMI distribution
    st.write('BMI Distribution')
    fig, ax = plt.subplots()
    sns.histplot(users_data['BMI'], kde=True, bins=20, ax=ax)
    st.pyplot(fig)


elif analysis_option == 'Weight and Height Trends':
    st.header('Weight and Height Trends Over Time')
    # Generate dummy time series data for weight and height trends
    time_series_data = pd.DataFrame({
        'Date': pd.date_range(start='2022-01-01', periods=365),
        'Weight_kg': np.random.uniform(50, 120, 365),
        'Height_cm': np.random.uniform(150, 200, 365)
    })
    time_series_data.set_index('Date', inplace=True)

    # Visualize weight and height trends over time
    st.write('Weight and Height Trends Over Time')
    st.line_chart(time_series_data)

# Repeat the same pattern for the remaining analysis options

elif analysis_option == 'Correlation Analysis':
    st.header('Correlation Between Weight/Height and Other Metrics')

    # Generate dummy data for other health-related metrics
    users_data['Fitness_Level'] = np.random.randint(1, 11, num_users)  # Dummy fitness level data (1-10 scale)
    users_data['Mental_Health_Score'] = np.random.randint(1, 11, num_users)  # Dummy mental health score data (1-10 scale)
    users_data['Diet_Habits'] = np.random.choice(['Healthy', 'Moderate', 'Unhealthy'], num_users)  # Dummy diet habits data

    # Calculate correlation matrix
    correlation_matrix = users_data[['Weight_kg', 'Height_cm', 'Fitness_Level', 'Mental_Health_Score']].corr()

    # Visualize correlation matrix
    st.write('Correlation Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


elif analysis_option == 'Goals Progress Tracking':
    st.header('Weight and Height Goals Progress Tracking')

    # Generate dummy data for goals progress tracking
    users_data['Weight_Goal_kg'] = np.random.uniform(50, 100, num_users)  # Dummy weight goal data
    users_data['Height_Goal_cm'] = np.random.uniform(160, 190, num_users)  # Dummy height goal data

    # Visualize goals progress tracking
    st.write('Weight and Height Goals Progress Tracking')
    fig, ax = plt.subplots()
    ax.scatter(users_data['Weight_kg'], users_data['Height_cm'], label='Actual Data')
    ax.scatter(users_data['Weight_Goal_kg'], users_data['Height_Goal_cm'], color='red', label='Goal Data')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Height (cm)')
    ax.legend()
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
    users_data['Health_Recommendations'] = np.random.choice(['Increase Physical Activity', 'Improve Diet Habits',
                                                             'Engage in Mindfulness Activities'], num_users)

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
