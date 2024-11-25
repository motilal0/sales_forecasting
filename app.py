import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Streamlit app title
st.title("Correlation & Forecasting")

# Upload the dataset
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Time Series Plots", "Correlation Analysis", "Forecasting"])

    # Tab 1: Time Series Plots
    with tab1:
        st.markdown("<h1 style='text-align: center; font-weight: bold; text-decoration: underline;'>Time Series Plots</h1>", unsafe_allow_html=True)

        # Create a copy of the original dataframe
        df_copy = df.copy()

        # Ensure the 'date' column is of datetime type (if not already)
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')  # Coerce invalid dates to NaT
        df_copy = df_copy.dropna(subset=['date'])  # Drop rows with invalid dates

        # Filter the data for the desired date range (May 2019 to June 2024)
        df_copy = df_copy[(df_copy['date'] >= '2019-05-01') & (df_copy['date'] <= '2024-06-30')]

        # Set the 'date' column as the DataFrame index
        df_copy.set_index('date', inplace=True)

        # Ensure the data is sorted by the index (dates)
        df_copy.sort_index(inplace=True)

        # Define the variables to plot
        variables = ['f2f', 'gdm', '1_1_email', '1_1_email_or', 'external_engagement', 'no_of_attendees', 'sales']

        # Create a subplot figure with one row per variable
        fig = make_subplots(
            rows=len(variables),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,  # Increase space between graphs (adjust this value as needed)
            subplot_titles=variables
        )

        # Add a trace for each variable
        for i, var in enumerate(variables):
            fig.add_trace(
                go.Scatter(
                    x=df_copy.index,
                    y=df_copy[var],
                    mode='lines+markers',
                    name=var,
                    line=dict(width=2),
                    marker=dict(size=5)
                ),
                row=i + 1,
                col=1
            )

        # Update layout for better aesthetics
        fig.update_layout(
            title={
                'text': "Time Series Plots of Variables",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=400 * len(variables),  # Dynamically adjust height based on the number of variables
            width=1000,  # Increase width to prevent label trimming
            showlegend=False,
            template="plotly_white",
            margin=dict(l=60, r=60, t=60, b=120)  # Add margins to prevent trimming of labels
        )

        # Update x-axis properties for all subplots (showing month labels on all rows)
        fig.update_xaxes(
            tickformat='%b %Y',  # Format as Month-Year (e.g., Jan 2020)
            tickangle=270,  # Change x-axis label angle to 75° (or 90° if you prefer)
            dtick="M3",  # Show ticks every 3 months
            tickmode="linear",  # Set tick mode to linear for proper spacing
            showgrid=True,  # Show gridlines for better readability
        )

        # Explicitly ensure x-axis labels are shown for all subplots
        for i in range(len(variables)):
            fig.update_xaxes(
                showticklabels=True,  # Make sure tick labels are visible on all subplots
                row=i + 1,
                col=1
            )

        # Update y-axis labels for each subplot
        for i, var in enumerate(variables):
            fig.update_yaxes(title_text=var, row=i + 1, col=1)

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    # Tab 2: Correlation Analysis
    with tab2:
        st.markdown("<h1 style='text-align: center; font-weight: bold; text-decoration: underline;'>Correlation Analysis</h1>", unsafe_allow_html=True)
        # Create a new dataframe excluding the 'date' column
        df_corr = df.drop(columns=['date'])

        # Ensure that the dataframe has no missing values
        df_corr = df_corr.dropna()

        # Calculate the correlation matrix
        corr_matrix = df_corr.corr()

        # Create the heatmap for the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

        # Set the title for the heatmap
        plt.title('Correlation Analysis')

        # Show the plot in Streamlit
        st.pyplot(plt)

        # Filter the correlation matrix for absolute correlations >= 0.2
        filtered_corr_matrix = corr_matrix[(corr_matrix.abs() >= 0.2)]

        # Extract variable pairs with significant correlation
        significant_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= 0.2:
                    significant_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    )

        # Print significant variable pairs
        st.write("### Variable Pairs with Significant Correlation:")
        for pair in significant_pairs:
            st.write(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
  
        # Feature importance
        # Create a new dataframe excluding the 'date' and 'sales' columns
        df_importance = df.drop(columns=['date', 'sales'])

        # Ensure there are no missing values
        df_importance = df_importance.dropna()

        # Define the target variable (sales)
        y = df['sales']

        # Define the features (excluding 'sales' and 'date')
        X = df_importance

        # Initialize the Random Forest Regressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_

        # Create a DataFrame for the feature importances
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

        # Sort the features by importance in descending order
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Pass the sorted data directly to the plot
        fig = go.Figure(data=[go.Bar(
            x=importance_df['Importance'],  # Importance values (already sorted)
            y=importance_df['Feature'],    # Feature names (already sorted)
            orientation='h',
            marker=dict(color=importance_df['Importance'], colorscale='Viridis'),
            text=importance_df['Importance'].round(4),  # Display importance values on the bars
            textposition='outside',  # Position the text outside the bars for visibility
        )])

        # Update the layout to add titles, labels, and customize the axes
        st.markdown("<h1 style='text-align: center; font-weight: bold; text-decoration: underline;'>Features Importance</h1>", unsafe_allow_html=True)
        fig.update_layout(
            title='Important Features for Predicting Sales',
            xaxis_title='Importance',
            yaxis_title='Features',
            template='plotly_white',
            xaxis=dict(showgrid=True, gridcolor='LightGray'),
            yaxis=dict(showgrid=False, categoryorder='total ascending'),  # Ensure order is maintained
            showlegend=False  # Hide the legend as it's unnecessary for this chart
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)


    # Tab 3: Forecasting
  
    with tab3:

        # Streamlit markdown header
        st.markdown("<h1 style='text-align: center; font-weight: bold; text-decoration: underline;'>Time Series Forecasting</h1>", unsafe_allow_html=True)

        # Data preprocessing
        df_preprocessed = df.copy()  # Create a copy to avoid overwriting the original
        df_preprocessed['date'] = pd.to_datetime(df_preprocessed['date'])
        df_preprocessed.set_index('date', inplace=True)
        df_preprocessed.sort_index(inplace=True)

        # Define exogenous variables
        exog_vars = ['f2f', 'gdm', '1_1_email', '1_1_email_or', 'external_engagement', 'no_of_attendees']
        df_exog = df_preprocessed[exog_vars]

        # ARIMAX model (Non-seasonal)
        model_arimax = ARIMA(df_preprocessed['sales'], exog=df_exog, order=(0, 1, 2))  # (p, d, q)
        model_arimax_fit = model_arimax.fit()

        # Forecast for the next 12 months
        forecast_arimax = model_arimax_fit.forecast(steps=12, exog=df_exog[-12:])
        forecast_dates = pd.date_range(df_preprocessed.index[-1], periods=13, freq='M')[1:]

        # Calculate MAPE for ARIMAX
        actual_sales = df_preprocessed['sales'][-12:]
        if len(actual_sales) == len(forecast_arimax):
            mape_arimax = np.mean(np.abs((actual_sales.values - forecast_arimax.values) / actual_sales.values)) * 100
        else:
            mape_arimax = np.nan

        # Create a Plotly figure for ARIMAX
        fig_arimax = go.Figure()
        fig_arimax.add_trace(go.Scatter(x=df_preprocessed.index, y=df_preprocessed['sales'], mode='lines', name='Historical Sales', line=dict(color='blue')))
        fig_arimax.add_trace(go.Scatter(x=forecast_dates, y=forecast_arimax, mode='lines', name='ARIMAX Forecast', line=dict(color='red', dash='dot')))
        fig_arimax.update_layout(title=f'ARIMAX Forecast<br>MAPE: {mape_arimax:.2f}%', xaxis_title='Date', yaxis_title='Sales', template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

        # SARIMAX model (Seasonal)
        model_sarimax = SARIMAX(df_preprocessed['sales'], exog=df_exog, order=(3, 1, 0), seasonal_order=(1, 1, 0, 12))  # Seasonal (p, d, q, s)
        model_sarimax_fit = model_sarimax.fit()

        # Forecast for the next 12 months
        forecast_sarimax = model_sarimax_fit.forecast(steps=12, exog=df_exog[-12:])
        forecast_dates = pd.date_range(df_preprocessed.index[-1], periods=13, freq='M')[1:]

        # Calculate MAPE for SARIMAX
        if len(actual_sales) == len(forecast_sarimax):
            mape_sarimax = np.mean(np.abs((actual_sales.values - forecast_sarimax.values) / actual_sales.values)) * 100
        else:
            mape_sarimax = np.nan

        # Create a Plotly figure for SARIMAX
        fig_sarimax = go.Figure()
        fig_sarimax.add_trace(go.Scatter(x=df_preprocessed.index, y=df_preprocessed['sales'], mode='lines', name='Historical Sales', line=dict(color='blue')))
        fig_sarimax.add_trace(go.Scatter(x=forecast_dates, y=forecast_sarimax, mode='lines', name='SARIMAX Forecast', line=dict(color='red', dash='dot')))
        fig_sarimax.update_layout(title=f'SARIMAX Forecast<br>MAPE: {mape_sarimax:.2f}%', xaxis_title='Date', yaxis_title='Sales', template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

        # Create a summary table for model performance
        model_performance = pd.DataFrame({
            'Model': ['ARIMAX', 'SARIMAX'],
            'MAPE': [mape_arimax, mape_sarimax],
            'AIC': [model_arimax_fit.aic, model_sarimax_fit.aic]
        })

        # Display the table
        st.write("### Model Comparison", model_performance)

        # Select the best model based on MAPE
        best_model_row = model_performance.loc[model_performance['MAPE'].idxmin()]
        best_model = best_model_row['Model']
        best_mape = best_model_row['MAPE']
        best_aic = best_model_row['AIC']
        best_forecast = forecast_arimax if best_model == "ARIMAX" else forecast_sarimax
        best_fig = fig_arimax if best_model == "ARIMAX" else fig_sarimax

        # Calculate the trend line using linear regression
        # Prepare data for trend line
        historical_dates = (df_preprocessed.index - df_preprocessed.index.min()).days.values.reshape(-1, 1)
        historical_sales = df_preprocessed['sales'].values

        # Fit a linear regression model
        trend_model = LinearRegression()
        trend_model.fit(historical_dates, historical_sales)

        # Generate trend line values for historical and forecasted dates
        all_dates = pd.concat([pd.Series(df_preprocessed.index), pd.Series(forecast_dates)])
        all_days = (all_dates - df_preprocessed.index.min()).dt.days.values.reshape(-1, 1)
        trend_values_all = trend_model.predict(all_days)

        # Add trend line to the best model figure
        best_fig.add_trace(go.Scatter(x=all_dates, y=trend_values_all, mode='lines', name='Trend Line', line=dict(color='green', dash='dash')))

        # Display the best model's details
        st.markdown(f"<h3 style='text-align: center; font-weight: bold; text-decoration: underline;'>Best Model Selected: {best_model}</h3>", unsafe_allow_html=True)
        st.write(f"Best Model: {best_model}")
        st.write(f"MAPE: {best_mape:.2f}%")
        st.write(f"Model AIC: {best_aic}")

        # Display the corresponding Plotly chart
        st.plotly_chart(best_fig)

        # Format the 'forecast_dates' to show month name and year (e.g., 'Jul 2024', 'Aug 2024')
        forecast_months = forecast_dates.strftime('%b %Y')

        # Create a DataFrame with 'Month' and 'Forecasted Sales' columns
        forecast_output = pd.DataFrame({
            'Month': forecast_months,
            'Forecasted Values': best_forecast.round(0)
        })

        # Remove index column explicitly by resetting the index
        forecast_output.reset_index(drop=True, inplace=True)
        st.write("### Forecasted Values:")
        st.dataframe(forecast_output)
