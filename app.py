import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import time
import matplotlib.pyplot as plt

# Load and prepare data
def load_data(file):
    """
    Load and preprocess the input data file
    
    Args:
        file: Uploaded CSV file
    
    Returns:
        Processed DataFrame
    """
    data = pd.read_csv(file)
    
    # Ensure date parsing
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Assuming 'Close' is the demand column - rename if different
    return data[['Date', 'Close']]

# Prepare data for LSTM
def prepare_dataframe_for_lstm(df, n_steps):
    """
    Prepare DataFrame for LSTM by creating lagged features
    
    Args:
        df: Input DataFrame
        n_steps: Number of lookback steps
    
    Returns:
        Prepared DataFrame with lagged features
    """
    df = dc(df)
    df.set_index('Date', inplace=True)
    
    # Create lagged features
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    
    df.dropna(inplace=True)
    return df

# LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        """
        Initialize LSTM model
        
        Args:
            input_size: Size of input features
            hidden_size: Number of hidden units
            num_stacked_layers: Number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass of the LSTM model
        
        Args:
            x: Input tensor
        
        Returns:
            Output predictions
        """
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def categorize_demand(demand):
    """
    Categorize demand into low, medium, and high segments
    
    Args:
        demand: Series of demand values
    
    Returns:
        Series of demand categories
    """
    low_threshold = demand.quantile(0.33)
    high_threshold = demand.quantile(0.66)
    
    categories = pd.cut(demand, 
        bins=[-float('inf'), low_threshold, high_threshold, float('inf')],
        labels=['Low Demand', 'Medium Demand', 'High Demand']
    )
    
    return categories

def main():
    """
    Main Streamlit application
    """
    # Streamlit App Configuration
    st.set_page_config(
        page_title="LSTM Demand Forecasting Dashboard", 
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )
    
    # Title and Description
    st.title("🚀 LSTM Demand Forecasting Dashboard")
    st.write("product demand prediction using Long Short-Term Memory (LSTM) Neural Networks")

    # Sidebar for Configuration
    st.sidebar.header("Model & Data Configuration")
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Product Demand CSV", 
        type="csv", 
        help="Upload a CSV file with Date and Close columns"
    )

    # Default Hyperparameters
    lookback = 7
    hidden_units = 4
    layers = 1

    if uploaded_file is not None:
        # Load Data
        data = load_data(uploaded_file)
        
        # Prepare Data for LSTM
        shifted_df = prepare_dataframe_for_lstm(data, lookback)
        
        # Data Scaling
        scaler = MinMaxScaler(feature_range=(-1, 1))
        shifted_df_as_np = scaler.fit_transform(shifted_df.to_numpy())
        
        X = shifted_df_as_np[:, 1:]
        y = shifted_df_as_np[:, 0]
        
        X = dc(np.flip(X, axis=1))
        X = X.reshape((-1, lookback, 1))
        
        # Convert to Tensor
        X_tensor = torch.tensor(X).float()
        
        # Model Loading & Prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTM(1, hidden_units, layers).to(device)
        
        try:
            model.load_state_dict(torch.load('lstm_model.pth'))
            model.eval()
        except FileNotFoundError:
            st.error("Pre-trained model not found. Please train the model first.")
            return

        # Predictions
        with torch.no_grad():
            predictions = model(X_tensor.to(device)).cpu().numpy()

        # Inverse Transform Predictions
        dummies = np.zeros((X_tensor.shape[0], lookback + 1))
        dummies[:, 0] = predictions.flatten()
        predictions_inverse = scaler.inverse_transform(dummies)[:, 0]

        # Forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': data['Date'].iloc[lookback:],
            'Predicted Demand': predictions_inverse
        })

        # Categorize Demand
        forecast_df['Demand Category'] = categorize_demand(forecast_df['Predicted Demand'])

        # Visualization Columns
        col1, col2, col3 = st.columns(3)

        with col1:
            # Demand Forecast Line Chart
            st.subheader("Demand Forecast Trend")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=forecast_df['Date'], 
                y=forecast_df['Predicted Demand'], 
                mode='lines+markers', 
                name='Predicted Demand',
                line=dict(color='red', width=2)
            ))
            fig_line.update_layout(
                xaxis_title="Date", 
                yaxis_title="Predicted Demand",
                template="plotly_white"
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with col2:
            # Distribution of Predicted Demands
            st.subheader("Demand Distribution")
            fig_hist = px.histogram(
                forecast_df, 
                x='Predicted Demand', 
                nbins=20, 
                title="Distribution of Predicted Demands"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col3:
            # Pie Chart of Demand Categories
            st.subheader("Demand Category Breakdown")
            demand_category_counts = forecast_df['Demand Category'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=demand_category_counts.index, 
                values=demand_category_counts.values,
                hole=0.3,
                marker_colors=['green', 'blue', 'red']
            )])
            fig_pie.update_layout(title_text="Demand Category Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Additional New Visualizations
        st.subheader("Advanced Demand Insights")
        col4, col5 = st.columns(2)

        with col4:
            # Box Plot of Demand Categories
            st.subheader("Demand Category Box Plot")
            fig_box = go.Figure()
            for category in forecast_df['Demand Category'].unique():
                category_data = forecast_df[forecast_df['Demand Category'] == category]['Predicted Demand']
                fig_box.add_trace(go.Box(y=category_data, name=category))
            fig_box.update_layout(title_text="Demand Distribution by Category")
            st.plotly_chart(fig_box, use_container_width=True)

        with col5:
            # Rolling Mean of Predicted Demand
            st.subheader("30-Day Rolling Mean Demand")
            rolling_mean = forecast_df['Predicted Demand'].rolling(window=30).mean()
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=forecast_df['Date'], 
                y=rolling_mean, 
                mode='lines', 
                name='30-Day Rolling Mean',
                line=dict(color='purple', width=2)
            ))
            fig_rolling.update_layout(
                xaxis_title="Date", 
                yaxis_title="Rolling Mean Demand",
                template="plotly_white"
            )
            st.plotly_chart(fig_rolling, use_container_width=True)

        # Additional Insights
        st.subheader("Forecast Insights")
        col6, col7, col8,col9 = st.columns(4)
        
        with col6:
            st.metric("Average Predicted Demand", 
                      f"{forecast_df['Predicted Demand'].mean():.2f}")
        
        with col7:
            st.metric("Maximum Predicted Demand", 
                      f"{forecast_df['Predicted Demand'].max():.2f}")
        
        with col8:
            st.metric("Minimum Predicted Demand", 
                      f"{forecast_df['Predicted Demand'].min():.2f}")
    
        
        # Price Distribution Pie Chart
        st.write("### Price Distribution of the Products (Pie Chart)")
        price_bins_str = pd.cut(data['Close'], bins=5).astype(str)
        pie_data = price_bins_str.value_counts().reset_index()
        pie_data.columns = ['Price Range', 'Count']
        pie_fig = px.pie(pie_data, names='Price Range', values='Count')
        st.plotly_chart(pie_fig)

if __name__ == "__main__":
    main()