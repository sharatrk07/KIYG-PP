import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/sharatrk/Desktop/SEM~2/OLYMP PREDICTION/OLYMP-FINAL.csv', sep=';')

position_mapping = {'FIRST': 1, 'SECOND': 2, 'THIRD': 3}

# Prepare features and target variable
X = data[['Sport Name', 'Event Name', 'State Name', 'Category']]
y = data['Position'].map(position_mapping)

# Define preprocessing and classification pipeline
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['Sport Name', 'Event Name', 'State Name', 'Category'])]
)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

def get_player_details(player_name, event_name):
    player_data = data[(data['First Name'] == player_name) & (data['Event Name'] == event_name)]
    if player_data.empty:
        return 'Please enter a valid player name or event name and try again', None

    features = player_data[['Sport Name', 'Event Name', 'State Name', 'Category']]
    features_transformed = pipeline.named_steps['preprocessor'].transform(features)
    predicted_position_scaled = pipeline.named_steps['classifier'].predict(features_transformed)[0]

    position_10_scale = predicted_position_scaled
    winning_percentage = max(0, min(100, 100 * (10 - predicted_position_scaled) / 9))

    return {
        'First Name': player_name,
        'Sport Name': player_data['Sport Name'].values[0],
        'Event Name': event_name,
        'State Name': player_data['State Name'].values[0],
        'Category': player_data['Category'].values[0],
        'Position': position_10_scale
    }, winning_percentage

# Sidebar navigation
st.sidebar.title("KIYG: Player Performance Prediction")
page = st.sidebar.radio("Go to", ["Participants list", "Prediction"])

# Main title
st.title("KHELO INDIA YOUTH GAMES")

if page == "Participants list":
    st.subheader("Participants List")
    player_event_df = data[['First Name', 'Sport Name', 'Event Name']].drop_duplicates()
    player_event_df.index = player_event_df.index + 1  # Start index from 1
    # Increase the height of the dataframe display
    st.dataframe(player_event_df, height=735)  # Adjust the height as needed

elif page == "Prediction":
    st.subheader("Player's Performance Prediction")

    # Initialize session state for player name and event name
    if 'player_name' not in st.session_state:
        st.session_state['player_name'] = ''
    if 'event_name' not in st.session_state:
        st.session_state['event_name'] = ''

    player_name_input = st.text_input('Enter player name:', value=st.session_state['player_name'])
    event_name_input = st.text_input('Enter event name:', value=st.session_state['event_name'])

    if st.button('Submit'):
        st.session_state['player_name'] = player_name_input
        st.session_state['event_name'] = event_name_input

        if player_name_input and event_name_input:
            details, winning_percentage = get_player_details(player_name_input, event_name_input)
            if isinstance(details, dict):
                st.write(f"**Details for {player_name_input} in the event {event_name_input}:**")
                st.write(f"**Sport Name**: {details['Sport Name']}")
                st.write(f"**Event Name**: {details['Event Name']}")
                st.write(f"**State Name**: {details['State Name']}")
                st.write(f"**Category**: {details['Category']}")
                st.markdown(f"**Predicted Position**: <span style='font-size:18px; color:green; font-weight:bold;'>{details['Position']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Winning Percentage**: <span style='font-size:18px; color:red; font-weight:bold;'>{winning_percentage:.2f}%</span>", unsafe_allow_html=True)

                # Display medal predictions
                if details['Position'] == 1:
                    st.markdown("**He is currently ranked first and has a high chance of winning the** <span style='color:gold; font-weight:bold;'>GOLD</span> **medal.**", unsafe_allow_html=True)
                elif details['Position'] == 2:
                    st.markdown("**He is currently ranked second and has a strong possibility of securing the** <span style='color:silver; font-weight:bold;'>SILVER</span> **medal.**", unsafe_allow_html=True)
                elif details['Position'] == 3:
                    st.markdown("**He is currently ranked third and is likely to win the** <span style='color:bronze; font-weight:bold;'>BRONZE</span> **medal.**", unsafe_allow_html=True)
            else:
                st.write(details)