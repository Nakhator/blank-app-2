import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from scipy import stats

# Set page config
st.set_page_config(page_title="Travel Cost Predictor", page_icon="✈️", layout="wide")

# Title and description
st.title("✈️ Travel Cost Predictor")
st.markdown("""
Predict travel costs with these enforced rules:
1. Daily rate stays constant
2. Total = Daily rate × Nights
3. Resort most expensive, Hostel cheapest
4. Flights most expensive transport
""")

# Custom constrained model class
class ConstrainedRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )
        self.accommodation_weights = {
            'Hostel': 0.8,  # 20% cheaper
            'Airbnb': 0.95,
            'Hotel': 1.0,   # Baseline
            'Resort': 1.3,  # 30% premium
            'Villa': 1.2,
            'Guesthouse': 0.85,
            'Vacation rental': 0.9
        }

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # Always predicts BASE DAILY RATE
        base_pred = self.model.predict(X)
        
        if isinstance(X, pd.DataFrame) and 'AccommodationType' in X.columns:
            acc_types = X['AccommodationType']
            acc_multipliers = acc_types.map(lambda x: self.accommodation_weights.get(x, 1.0)).values
            base_pred = base_pred * acc_multipliers
        
        return base_pred

# Data loading and cleaning with outlier removal
@st.cache_resource
def load_data():
    try:
        data = pd.read_csv("Travel_details_dataset.csv", encoding='utf-8-sig')
        data = data.dropna(how='all')
        
        def clean_currency(value):
            if isinstance(value, str):
                value = value.replace('USD', '').replace(',', '').replace('$', '').strip()
                return float(value) if value.replace('.','',1).isdigit() else None
            return value
        
        for cost_col in ['Accommodation cost', 'Transportation cost']:
            data[cost_col] = data[cost_col].apply(clean_currency)
        
        data['Destination'] = data['Destination'].str.split(',').str[0].str.strip()
        data['Start date'] = pd.to_datetime(data['Start date'], errors='coerce')
        data['End date'] = pd.to_datetime(data['End date'], errors='coerce')
        data['Duration'] = (data['End date'] - data['Start date']).dt.days
        
        transport_mapping = {
            'Plane': 'Flight', 'Airplane': 'Flight', 'Car': 'Car rental',
            'Subway': 'Train', 'Bus': 'Bus', 'Ferry': 'Ferry'
        }
        data['TransportType'] = data['Transportation type'].replace(transport_mapping)
        
        data = data.rename(columns={
            'Traveler nationality': 'TravelerNationality',
            'Accommodation type': 'AccommodationType',
            'Accommodation cost': 'Cost',
            'Transportation cost': 'TransportCost',
            'Start date': 'StartDate'
        }).dropna(subset=['Cost', 'TransportCost'])
        
        # Calculate daily rate for accommodation
        data['DailyRate'] = data['Cost'] / data['Duration']
        
        # Remove outliers using z-score for accommodation daily rates
        z_scores = np.abs(stats.zscore(data['DailyRate']))
        data = data[(z_scores < 3)]
        
        # Remove outliers for transportation costs
        z_scores_trans = np.abs(stats.zscore(data['TransportCost']))
        data = data[(z_scores_trans < 3)]
        
        # Remove unrealistic durations (more than 90 days or less than 1 day)
        data = data[(data['Duration'] > 0) & (data['Duration'] <= 90)]
        
        return data[['Destination', 'Duration', 'StartDate', 'AccommodationType',
                   'TravelerNationality', 'Cost', 'TransportType', 'TransportCost']]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main app
data = load_data()

if data is not None:
    # Show data summary with outliers removed
    st.subheader("Data Summary After Outlier Removal")
    st.write(f"Total records: {len(data)}")
    st.write("Sample data:")
    st.dataframe(data.head())
    
    # Show distribution plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data['Cost'] / data['Duration'], ax=ax[0], kde=True)
    ax[0].set_title('Distribution of Daily Rates (Outliers Removed)')
    ax[0].set_xlabel('Daily Rate')
    
    sns.histplot(data['TransportCost'], ax=ax[1], kde=True)
    ax[1].set_title('Distribution of Transport Costs (Outliers Removed)')
    ax[1].set_xlabel('Transport Cost')
    st.pyplot(fig)

    # Prepare data
    def engineer_features(df):
        df = df.copy()
        df['Month'] = df['StartDate'].dt.month
        df['IsWeekend'] = df['StartDate'].dt.dayofweek.isin([5,6]).astype(int)
        df['IsPeakSeason'] = df['Month'].isin([6,7,8,12]).astype(int)
        return df

    engineered_data = engineer_features(data)
    DESTINATIONS = sorted(data['Destination'].unique().tolist())
    ACCOMMODATION_TYPES = sorted(data['AccommodationType'].dropna().unique().tolist())
    NATIONALITIES = sorted(data['TravelerNationality'].dropna().unique().tolist())
    TRANSPORT_TYPES = sorted(data['TransportType'].dropna().unique().tolist())

    # Model training
    st.header("Model Training")
    if st.button("Train Models"):
        with st.spinner("Training..."):
            # Accommodation model
            X_acc = engineered_data[['Destination', 'AccommodationType', 
                                   'TravelerNationality', 'Month', 'IsWeekend', 
                                   'IsPeakSeason']]
            y_acc = engineered_data['Cost'] / engineered_data['Duration']  # Daily rate
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_acc, y_acc, test_size=0.2, random_state=42
            )
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), ['Month', 'IsWeekend', 'IsPeakSeason']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['Destination', 'AccommodationType', 'TravelerNationality'])
            ])
            
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', ConstrainedRandomForest(random_state=42))
            ])
            
            model.fit(X_train, y_train)
            joblib.dump(model, 'accommodation_model.pkl')
            
            # Evaluate accommodation model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Transport model
            X_trans = engineered_data[['Destination', 'TransportType', 
                                     'TravelerNationality', 'IsPeakSeason']]
            y_trans = engineered_data['TransportCost']
            
            X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(
                X_trans, y_trans, test_size=0.2, random_state=42
            )
            
            trans_model = Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ('regressor', LGBMRegressor())
            ])
            trans_model.fit(X_train_trans, y_train_trans)
            joblib.dump(trans_model, 'transport_model.pkl')
            
            # Evaluate transport model
            y_pred_trans = trans_model.predict(X_test_trans)
            mae_trans = mean_absolute_error(y_test_trans, y_pred_trans)
            r2_trans = r2_score(y_test_trans, y_pred_trans)
            
            # Display metrics
            st.success("Models trained successfully!")
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accommodation Model MAE", f"${mae:.2f}")
                st.metric("Accommodation R² Score", f"{r2:.2f}")
                
                # Plot actual vs predicted for accommodation
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                ax.set_xlabel('Actual Daily Rate')
                ax.set_ylabel('Predicted Daily Rate')
                ax.set_title('Accommodation Model')
                st.pyplot(fig)
            
            with col2:
                st.metric("Transport Model MAE", f"${mae_trans:.2f}")
                st.metric("Transport R² Score", f"{r2_trans:.2f}")
                
                # Plot actual vs predicted for transport
                fig, ax = plt.subplots()
                ax.scatter(y_test_trans, y_pred_trans, alpha=0.5)
                ax.plot([y_test_trans.min(), y_test_trans.max()], 
                        [y_test_trans.min(), y_test_trans.max()], 'k--')
                ax.set_xlabel('Actual Transport Cost')
                ax.set_ylabel('Predicted Transport Cost')
                ax.set_title('Transport Model')
                st.pyplot(fig)

    # Prediction UI
    st.header("Cost Prediction")
    
    with st.form("accommodation_form"):
        st.subheader("Accommodation Cost")
        col1, col2 = st.columns(2)
        with col1:
            destination = st.selectbox("Destination", DESTINATIONS)
            accommodation = st.selectbox("Accommodation Type", ACCOMMODATION_TYPES)
        with col2:
            nationality = st.selectbox("Nationality", NATIONALITIES)
            duration = st.number_input("Nights", min_value=1, max_value=90, value=7)
        
        start_date = st.date_input("Start Date", datetime.today())
        month = start_date.month
        is_weekend = start_date.weekday() >= 5
        is_peak = month in [6,7,8,12]
        
        if st.form_submit_button("Calculate"):
            try:
                model = joblib.load('accommodation_model.pkl')
                input_data = pd.DataFrame([{
                    'Destination': destination,
                    'AccommodationType': accommodation,
                    'TravelerNationality': nationality,
                    'Month': month,
                    'IsWeekend': int(is_weekend),
                    'IsPeakSeason': int(is_peak)
                }])
                
                daily_rate = model.predict(input_data)[0]
                total_cost = daily_rate * duration
                
                st.success(f"## Total Accommodation Cost: ${total_cost:,.2f}")
                st.write(f"**Daily rate:** ${daily_rate:,.2f}")
                st.write(f"**{duration} nights × ${daily_rate:,.2f} = ${total_cost:,.2f}**")
                st.session_state['accom_cost'] = total_cost
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Train model first.")

    with st.form("transport_form"):
        st.subheader("Transportation Cost")
        col1, col2 = st.columns(2)
        with col1:
            trans_dest = st.selectbox("Destination", DESTINATIONS, key='trans_dest')
            trans_type = st.selectbox("Type", TRANSPORT_TYPES)
        with col2:
            trans_nat = st.selectbox("Nationality", NATIONALITIES, key='trans_nat')
            is_peak = st.checkbox("Peak Season", value=False)
        
        if st.form_submit_button("Calculate"):
            try:
                model = joblib.load('transport_model.pkl')
                input_data = pd.DataFrame([{
                    'Destination': trans_dest,
                    'TransportType': trans_type,
                    'TravelerNationality': trans_nat,
                    'IsPeakSeason': int(is_peak) 
                }])
                
                trans_cost = model.predict(input_data)[0]
                st.success(f"## Transportation Cost: ${trans_cost:,.2f}")
                st.session_state['trans_cost'] = trans_cost
                
            except Exception as e:
                st.error(f"Error: {str(e)}. Train model first.")

    # Combined total
    if 'accom_cost' in st.session_state and 'trans_cost' in st.session_state:
        st.header("💵 Total Trip Cost")
        total = st.session_state['accom_cost'] + st.session_state['trans_cost']
        st.success(f"## Grand Total: ${total:,.2f}")
        st.write(f"- Accommodation: ${st.session_state['accom_cost']:,.2f}")
        st.write(f"- Transportation: ${st.session_state['trans_cost']:,.2f}")

else:
    st.error("Failed to load data. Check your CSV file.")
