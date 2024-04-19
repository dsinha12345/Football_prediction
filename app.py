from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

df=pd.read_csv('E0.csv')

rows=(df.columns)

df["mean_of_betting_home"] = df[["B365H","BWH","IWH","PSH","WHH","VCH"]].mean(axis=1)
df["mean_of_betting_draw"] = df[["B365D","BWD","IWD","PSD","WHD","VCD"]].mean(axis=1)
df["mean_of_betting_away"] = df[["B365A","BWA","IWA","PSA","WHA","VCA"]].mean(axis=1)

df1= df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC',
       'AC', 'HY', 'AY', 'HR', 'AR',"mean_of_betting_home","mean_of_betting_draw","mean_of_betting_away"]]



df1.drop(columns = ["Div","Date","Referee","HTR"],inplace=True)

df1[['A', 'D', 'H']] = pd.get_dummies(df1['FTR'])

df1['FTR'] = df1.apply(lambda row: row['AwayTeam'] if row['A'] == 1 else (row['HomeTeam'] if row['H'] == 1 else 'Draw'), axis=1)

def update_third_column(row):
    if row['FTR'] == row['HomeTeam']:
        row["New"] = np.eye(3)[0]
    elif row['FTR'] == row['AwayTeam']:
        row["New"] = np.eye(3)[2]
    else:
        row["New"] = np.eye(3)[1]
    return row

# Apply the function to update the third column
df1 = df1.apply(update_third_column, axis=1)

label_mapping = {'Arsenal': 0, 'Aston Villa': 1, 'Bournemouth': 2, 'Brentford': 3, 'Brighton': 4, 'Burnley': 5, 'Cardiff': 6, 'Chelsea': 7, 'Crystal Palace': 8, 'Draw': 9, 'Everton': 10, 'Fulham': 11, 'Huddersfield': 12, 'Hull': 13, 'Leeds': 14, 'Leicester': 15, 'Liverpool': 16, 'Luton': 17, 'Man City': 18, 'Man United': 19, 'Middlesbrough': 20, 'Newcastle': 21, 'Norwich': 22, "Nott'm Forest": 23, 'QPR': 24, 'Sheffield United': 25, 'Southampton': 26, 'Stoke': 27, 'Sunderland': 28, 'Swansea': 29, 'Tottenham': 30, 'Watford': 31, 'West Brom': 32, 'West Ham': 33, 'Wolves': 34}
teams = list(label_mapping.keys())
teams.pop(9)

df1['HomeTeam'] = df1['HomeTeam'].replace(label_mapping)
df1['AwayTeam'] = df1['AwayTeam'].replace(label_mapping)
df1['FTR'] = df1['FTR'].replace(label_mapping)

if 'FTR' in df1.columns:
    # Drop the 'FTR' column inplace
    df1.drop(["FTR","A","D","H"], axis=1, inplace=True)
    print("Column 'FTR' dropped successfully.")
else:
    print("Column 'FTR' does not exist in the DataFrame.")

X = df1.drop('New', axis=1)
y = df1['New'].apply(pd.Series)

col_to_scale = ['HS','AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR','mean_of_betting_home', 'mean_of_betting_draw', 'mean_of_betting_away']
scaler = StandardScaler()
df1[col_to_scale] = scaler.fit_transform(df1[col_to_scale])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f'Test accuracy: {test_acc}')

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def rename_team(row):
    for k, v in label_mapping.items():
        if row["HomeTeam"] == v:
            hometeam = k
        if row["AwayTeam"] == v:
            awayteam = k
    if row["New"][0] == 1:
        row["New"] = hometeam
    elif row["New"][2] == 1:
        row["New"] = awayteam
    else:
        row["New"] = "Draw"
    row["HomeTeam"] = hometeam
    row["AwayTeam"] = awayteam

    return row

def predict_winner(home_team, away_team):
    home_name=home_team
    away_name=away_team

    for k,v in label_mapping.items():
        if home_name == k:
            home_name = v
        if away_name == k:
            away_name = v

    print(home_name,away_name)

    features = ['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC',
       'AC', 'HY', 'AY', 'HR', 'AR', 'mean_of_betting_home',
       'mean_of_betting_draw', 'mean_of_betting_away']

    filtered_df = df1[(df1['HomeTeam'] == label_mapping[home_team]) & (df1['AwayTeam'] == label_mapping[away_team])]
    print("Previous Stats")
    print(filtered_df)
    match_data = filtered_df[features].mean().values.reshape(1, -1)

    result = model.predict(match_data)
    prediction_result = f"There is a {round(result[0, 0] * 100, 2)}% chance {home_team} will win<br>" \
                        f"There is a {round(result[0, 2] * 100, 2)}% chance {away_team} will win<br>" \
                f"There is a {round(result[0, 1] * 100, 2)}% chance it will be a draw"
    previous_df = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]


    column_mapping = {
    'HS': 'Home Shots',
    'AS': 'Away Shots',
    'HST': 'Home Shots on Target',
    'AST': 'Away Shots on Target',
    'HF': 'Home Fouls',
    'AF': 'Away Fouls',
    'HC': 'Home Corners',
    'AC': 'Away Corners',
    'HY': 'Home Yellow Cards',
    'AY': 'Away Yellow Cards',
    'HR': 'Home Red Cards',
    'AR': 'Away Red Cards',
    "FTR" : "Full Time Result"
}

    columns_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'Full Time Result', 'Referee',
                   'Home Shots', 'Away Shots', 'Home Shots on Target', 'Away Shots on Target',
                   'Home Fouls', 'Away Fouls', 'Home Corners', 'Away Corners',
                   'Home Yellow Cards', 'Away Yellow Cards', 'Home Red Cards', 'Away Red Cards']

    previous_df.rename(columns=column_mapping, inplace=True)
    return prediction_result,previous_df.filter(columns_to_keep)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filtered_df = None
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        result, filtered_df = predict_winner(home_team, away_team)
    return render_template('index.html', teams=teams, result=result,filtered_df = filtered_df)


if __name__ == '__main__':
    app.run(debug=True)