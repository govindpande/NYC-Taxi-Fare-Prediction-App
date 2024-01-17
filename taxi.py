# Import necessary libraries
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Packages for OLS, MLR, confusion matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics # For confusion matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
file_path = '2017_Yellow_Taxi_Trip_Data.csv'
taxi_data = pd.read_csv(file_path)


    # Impute duration and create a new column
taxi_data['tpep_pickup_datetime'] = pd.to_datetime(taxi_data['tpep_pickup_datetime'])
taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(taxi_data['tpep_dropoff_datetime'])
taxi_data['duration'] = (taxi_data['tpep_dropoff_datetime'] - taxi_data['tpep_pickup_datetime']).dt.total_seconds()


# Streamlit app header
st.title('NYC Taxi Data Analysis App')

# Navigation
page = st.sidebar.selectbox("Choose a page", ["Analysis", "ML Prediction"])

# Analysis Page
if page == "Analysis":
    # Display basic information about the dataset
    st.subheader('Please navigate to Prediction page using side navigation.')

    # Display the first few rows of the dataset
    st.subheader('Sample Data:')
    st.write(taxi_data.head())

    # Impute duration and create a new column
    taxi_data['tpep_pickup_datetime'] = pd.to_datetime(taxi_data['tpep_pickup_datetime'])
    taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(taxi_data['tpep_dropoff_datetime'])
    taxi_data['duration'] = (taxi_data['tpep_dropoff_datetime'] - taxi_data['tpep_pickup_datetime']).dt.total_seconds()

    # Example: Trip distance distribution
    st.subheader('Distribution of Trip Distance:')
    fig_trip_distance = px.histogram(taxi_data, x='trip_distance', nbins=50,
                                     labels={'trip_distance': 'Trip Distance (miles)', 'count': 'Frequency'})
    st.plotly_chart(fig_trip_distance)

    # Example: Monthly ride counts
    taxi_data['pickup_month'] = taxi_data['tpep_pickup_datetime'].dt.month
    monthly_ride_counts = taxi_data['pickup_month'].value_counts().sort_index().reset_index()
    monthly_ride_counts.columns = ['Month', 'Ride Count']

    st.subheader('Monthly Ride Counts in 2017:')
    fig_monthly_ride_counts = px.bar(monthly_ride_counts, x='Month', y='Ride Count',
                                     labels={'Month': 'Month', 'Ride Count': 'Number of Rides'})
    st.plotly_chart(fig_monthly_ride_counts)

    


# ML Prediction Page
elif page == "ML Prediction":
    # Feature Engineering
    # Create 'rush_hour' column
    taxi_data['tpep_pickup_datetime'] = pd.to_datetime(taxi_data['tpep_pickup_datetime'])
    taxi_data['day'] = taxi_data['tpep_pickup_datetime'].dt.day_name()
    taxi_data['rush_hour'] = taxi_data['tpep_pickup_datetime'].dt.hour





    
    # If day is Saturday or Sunday, impute 0 in `rush_hour` column
    taxi_data.loc[taxi_data['day'].isin(['Saturday', 'Sunday']), 'rush_hour'] = 0

    def rush_hourizer(hour):
        if 6 <= hour['rush_hour'] < 10:
            val = 1
        elif 16 <= hour['rush_hour'] < 20:
            val = 1
        else:
            val = 0
        return val
    # Apply the `rush_hourizer()` function to the new column
    ### YOUR CODE HERE ###
    taxi_data.loc[(taxi_data.day != 'saturday') & (taxi_data.day != 'sunday'), 'rush_hour'] = taxi_data.apply(rush_hourizer, axis=1)
    # Feature Selection
    # Assuming 'VendorID', 'passenger_count', 'fare_amount', 'trip_distance', 'duration' columns are available
    selected_features = ['VendorID', 'passenger_count', 'fare_amount', 'trip_distance', 'duration']

    # Create a DataFrame with selected features
    ml_data = taxi_data[selected_features + ['rush_hour']].copy()


    # Plot the scatter matrix for machine learning
    st.subheader('Scatter Matrix for Machine Learning:')
    numerical_columns = ml_data.select_dtypes(include=['number']).columns
    scatter_matrix = px.scatter_matrix(ml_data, dimensions=numerical_columns)
    st.plotly_chart(scatter_matrix)

    # Create correlation heatmap with dark background using pearson correlation
    st.subheader('Correlation Heatmap for Machine Learning:')
    corr = ml_data.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)


    # Remove the target column fare amount from the data
    X = ml_data.drop(columns=['fare_amount'])

    y = ml_data[['fare_amount']]

    X['VendorID'] = X['VendorID'].astype(str)

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize the X variables
    ### YOUR CODE HERE ###
    # Standardize the X variables
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    print('X_train scaled:', X_train_scaled)

    
    # Fit your model to the training data
    ### YOUR CODE HERE ###
    lr=LinearRegression()
    lr.fit(X_train_scaled, y_train)


    # Evaluate the model performance on the training data in Streamlit
    st.subheader('Model Performance on Training Data:')

    # Coefficient of determination (R^2)
    r_sq = lr.score(X_train_scaled, y_train)
    st.write(f'Coefficient of determination (R^2): {r_sq:.4f}')

    # R-squared
    y_pred_train = lr.predict(X_train_scaled)
    r2 = r2_score(y_train, y_pred_train)
    st.write(f'R-squared (R2): {r2:.4f}')

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_train, y_pred_train)
    st.write(f'Mean Absolute Error (MAE): {mae:.4f}')

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_train, y_pred_train)
    st.write(f'Mean Squared Error (MSE): {mse:.4f}')

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    X_test_scaled = scaler.transform(X_test)

    # Evaluate the model performance on the testing data
    r_sq_test = lr.score(X_test_scaled, y_test)
    st.write(f'Coefficient of determination (R^2) on Testing Data: {r_sq_test:.4f}')

    y_pred_test = lr.predict(X_test_scaled)

    # R-squared on Testing Data
    r2_test = r2_score(y_test, y_pred_test)
    st.write(f'R-squared (R2) on Testing Data: {r2_test:.4f}')

    # Mean Absolute Error (MAE) on Testing Data
    mae_test = mean_absolute_error(y_test, y_pred_test)
    st.write(f'Mean Absolute Error (MAE) on Testing Data: {mae_test:.4f}')

    # Mean Squared Error (MSE) on Testing Data
    mse_test = mean_squared_error(y_test, y_pred_test)
    st.write(f'Mean Squared Error (MSE) on Testing Data: {mse_test:.4f}')

    # Root Mean Squared Error (RMSE) on Testing Data
    rmse_test = np.sqrt(mse_test)
    st.write(f'Root Mean Squared Error (RMSE) on Testing Data: {rmse_test:.4f}')



    # Feature Engineering
    # Assuming 'VendorID', 'passenger_count', 'fare_amount', 'trip_distance', 'duration' columns are available
    selected_features = ['VendorID', 'passenger_count', 'trip_distance', 'duration']
    X = ml_data[selected_features]
    y = ml_data[['fare_amount']]

    # Remove the 'fare_amount' column from the data
    #X = X.drop(columns=['fare_amount'])

    # Convert 'VendorID' to string and apply one-hot encoding
    X['VendorID'] = X['VendorID'].astype(str)
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize the X variables
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Train the linear regression model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Add a section for Predictor
    st.sidebar.title('Predictor')

    # Add user input widgets for predictor variables
    st.sidebar.header('Input Predictor Variables')

    # Change Vendor ID input to text
    vendor_id = st.sidebar.text_input('Vendor ID',1)

    # Ensure slider values have the appropriate types (float64)
    passenger_count = st.sidebar.slider('Passenger Count', float(X['passenger_count'].min()), float(X['passenger_count'].max()), float(X['passenger_count'].mean()))
    trip_distance = st.sidebar.slider('Trip Distance', float(X['trip_distance'].min()), float(X['trip_distance'].max()), float(X['trip_distance'].mean()))
    duration = st.sidebar.slider('Duration',1,240)

    # Create a DataFrame with user inputs
    user_input = pd.DataFrame({
        'VendorID': [vendor_id],
        'passenger_count': [passenger_count],
        'trip_distance': [trip_distance],
        'duration': [duration]
    })

    # Apply one-hot encoding to user input
    user_input['VendorID'] = user_input['VendorID'].astype(str)
    user_input = pd.get_dummies(user_input, drop_first=True)

    # Ensure the user input has the same features as the training data
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    # Standardize the user inputs using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict the target variable
    prediction = lr.predict(user_input_scaled)

    # Display the prediction in the Streamlit app
    st.sidebar.header('Prediction')
    st.sidebar.write(f'The predicted fare amount is: ${prediction[0][0]:.2f}')


    # Create a `results` dataframe
    ### YOUR CODE HERE ###
    results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                                'predicted': y_pred_test.ravel()})
    results['residual'] = results['actual'] - results['predicted']
    results.head()

    # Display the scatterplot in Streamlit
    st.subheader('Scatterplot: Actual vs. Predicted')

    # Create a scatterplot to visualize `predicted` over `actual`
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.set(style='whitegrid')
    sns.scatterplot(x='actual',
                    y='predicted',
                    data=results,
                    s=20,
                    alpha=0.5,
                    ax=ax
    )
    # Draw an x=y line to show what the results would be if the model were perfect
    plt.plot([results['actual'].min(), results['actual'].max()], [results['actual'].min(), results['actual'].max()], c='red', linewidth=2)
    plt.title('Actual vs. Predicted')

    # Display the plot in the Streamlit app
    st.pyplot(fig)


    # Display the distribution of residuals in Streamlit
    st.subheader('Distribution of Residuals')

    # Visualize the distribution of the residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(results['residual'], bins=np.arange(-15, 15.5, 0.5), ax=ax)
    plt.title('Distribution of the Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Count')

    # Display the plot in the Streamlit app
    st.pyplot(fig)
