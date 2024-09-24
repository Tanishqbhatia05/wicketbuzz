import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Declaring the teams
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# Declaring the venues where the matches are going to take place
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Loading our machine learning model from a saved pickle file
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Setting up the app's title
st.title('IPL Win Predictor')

# Setting up the layout with two columns
col1, col2 = st.columns(2)

# Creating a dropdown selector for the batting team
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))

# Creating a dropdown selector for the bowling team
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

# Creating a dropdown selector for the city where the match is being played
city = st.selectbox('Select the city where the match is being played', sorted(cities))

# Creating a numeric input for the target score
target = st.number_input('Target', min_value=0, max_value=300, step=1)

# Setting up the layout with three columns
col3, col4, col5 = st.columns(3)

# Creating a numeric input for the current score
with col3:
    score = st.number_input('Score', min_value=0, step=1)

# Creating a numeric input for the number of overs completed
with col4:
    overs = st.number_input('Overs Completed', min_value=0, max_value=20, step=1)

# Creating a numeric input for the number of wickets fallen
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# Checking for different match results based on the input provided
if score > target:
    st.write(battingteam, "won the match")
elif score == target - 1 and overs == 20:
    st.write("Match Drawn")
elif wickets == 10 and score < target - 1:
    st.write(bowlingteam, 'Won the match')
elif wickets == 10 and score == target - 1:
    st.write('Match tied')
elif battingteam == bowlingteam:
    st.write('To proceed, please select different teams because no match can be played between the same teams')
else:
    # Checking if the input values are valid
    if 0 <= target <= 300 and 0 <= overs <= 20 and 0 <= wickets <= 10 and score >= 0:
        if st.button('Predict Probability'):
            try:
                # Calculating the number of runs left for the batting team to win
                runs_left = target - score
                # Calculating the number of balls left 
                balls_left = 120 - (overs * 6)
                # Calculating the number of wickets left for the batting team
                wickets_left = 10 - wickets
                # Calculating the current Run-Rate of the batting team
                current_run_rate = score / overs if overs > 0 else 0
                # Calculating the Required Run-Rate for the batting team to win
                required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0

                # Creating a pandas DataFrame containing the user inputs
                input_df = pd.DataFrame({
                    'batting_team': [battingteam], 
                    'bowling_team': [bowlingteam], 
                    'city': [city], 
                    'runs_left': [runs_left], 
                    'balls_left': [balls_left],
                    'wickets': [wickets_left], 
                    'total_runs_x': [target], 
                    'cur_run_rate': [current_run_rate], 
                    'req_run_rate': [required_run_rate]
                })

                # Loading the trained machine learning pipeline to make the prediction
                result = pipe.predict_proba(input_df)
                
                # Extracting the likelihood of loss and win
                lossprob = result[0][0]
                winprob = result[0][1]

                # Displaying the predicted likelihood of winning and losing in percentage
                st.header(f"{battingteam} - {round(winprob * 100)}%")
                st.header(f"{bowlingteam} - {round(lossprob * 100)}%")
                
            except ZeroDivisionError:
                st.error("Please fill all the details correctly")
    else:
        st.error('There is something wrong with the input, please fill the correct details')
