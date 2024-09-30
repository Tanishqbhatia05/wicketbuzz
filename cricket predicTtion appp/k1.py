import streamlit as st
import pandas as pd
import pickle

# Declaring the teams
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

# Declaring the venues
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the pre-trained model pipeline
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception as e:
    st.error("Error loading the model. Please check if 'pipe.pkl' is available.")
    st.stop()

st.title('IPL Win Predictor')

# Team selection columns
col1, col2 = st.columns(2)
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

# City selection
city = st.selectbox('Select the city where the match is being played', sorted(cities))

# Input fields
target = st.number_input('Target', step=1, min_value=0)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', step=1, min_value=0)
with col4:
    overs = st.number_input('Overs Completed', step=1, min_value=0)
with col5:
    wickets = st.number_input('Wickets Fallen', step=1, min_value=0)

# Match result conditions
if score > target:
    st.write(battingteam, "won the match")
elif score == target - 1 and overs == 20:
    st.write("Match Drawn")
elif wickets == 10 and score < target - 1:
    st.write(bowlingteam, 'won the match')
elif battingteam == bowlingteam:
    st.write('To proceed, please select different teams as no match can be played between the same teams')
else:
    if 0 <= target <= 300 and 0 <= overs <= 20 and 0 <= wickets <= 10 and score >= 0:
        if st.button('Predict Probability'):
            try:
                runs_left = target - score
                balls_left = 120 - (overs * 6)
                wickets_remaining = 10 - wickets
                
                # Handle division by zero for current run rate and required run rate
                current_run_rate = score / overs if overs > 0 else 0
                required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

                # Prepare the input DataFrame with the correct column names
                input_df = pd.DataFrame({
                    'batting_team': [battingteam],
                    'bowling_team': [bowlingteam],
                    'city': [city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets_remaining],  # Changed to 'wickets_left'
                    'total_runs_x': [target],
                    'cur_run_rate': [current_run_rate],
                    'req_run_rate': [required_run_rate]
                })

                # Making predictions
                result = pipe.predict_proba(input_df)
                loss_prob = result[0][0]
                win_prob = result[0][1]

                st.header(f"{battingteam} - {round(win_prob * 100)}%")
                st.header(f"{bowlingteam} - {round(loss_prob * 100)}%")
            except Exception as e:
                st.error("An error occurred during prediction. Please check your inputs.")
                st.write(e)
    else:
        st.error('Please ensure all inputs are within the correct range as per IPL T-20 format.')
