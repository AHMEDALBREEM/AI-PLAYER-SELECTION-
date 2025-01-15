import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import random
from io import StringIO


POSITIONS = {
    "GK": "Goalkeeper",
    "LB": "Left Back",
    "RB": "Right Back",
    "CB": "Center Back",
    "CDM": "Central Defensive Midfielder",
    "CM": "Central Midfielder",
    "CAM": "Central Attacking Midfielder",
    "LM": "Left Midfielder",
    "RM": "Right Midfielder",
    "LW": "Left Winger",
    "RW": "Right Winger",
    "ST": "Striker",
}


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        if 'eval_metric' in kwargs:
            kwargs.pop('eval_metric')
        self.model = xgb.XGBClassifier(**kwargs, eval_metric='logloss')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
         self.model.set_params(**params)
         return self

def select_players(team_name, match_date):
    # Read CSV files
    team_df = pd.read_csv('team.csv')
    sub_player_df = pd.read_csv('sub-player.csv')
    warm_up_df = pd.read_csv('warm-up.csv')
    game_timetable_df = pd.read_csv('game-timetable.csv')
    player_data_df = pd.read_csv('player_data.csv')
    game_history_df = pd.read_csv('game_history.csv')
    
    # Filter for the given team and match date
    team_players = team_df[team_df['team_name'] == team_name].copy()
    match_details = game_timetable_df[
        (game_timetable_df['team_name'] == team_name) &
        (game_timetable_df['match_date'] == match_date)
    ]
    
    if match_details.empty:
        raise ValueError("Match details not found. Please check the input data.")
    
    # Rule 1: Players must be available in the team.csv
    eligible_players = team_players.copy()
    eligible_players.loc[:, 'anti_action'] = (eligible_players['defensive_skill'] + eligible_players['offensive_skill'])/2
    # Rule 2: Players must also exist in sub-player.csv
    eligible_players = eligible_players[eligible_players['name'].isin(sub_player_df['name'])]

    # Rule 3: Players must also exist in warm-up.csv
    eligible_players = eligible_players[eligible_players['name'].isin(warm_up_df['name'])]

    # Add ML prediction here
    selected_players, plan, player_positions = select_players_with_ml(eligible_players, game_timetable_df, team_name, player_data_df, game_history_df)

    return selected_players, plan, player_positions



def select_players_with_ml(eligible_players, game_timetable_df, team_name, player_data_df, game_history_df):
    # Load historical player data for training the ML model
    
    # Add new features
    player_data_df = add_new_features(player_data_df, game_timetable_df)
    
    # Features: power, match_frequency, experience, defensive_skill, offensive_skill, avg_goals, avg_assists, coach_rating
    X = player_data_df[['power', 'match_frequency', 'experience', 'defensive_skill', 'offensive_skill', 'avg_goals', 'avg_assists', 'coach_rating']]
    y = player_data_df['selected']  # Target variable: whether the player was selected (1: Yes, 0: No)

        # Check if there is sufficient data
    if len(X) < 10:
         raise ValueError("Insufficient training data. Please provide more training samples.")

    # Handle imbalanced data
    X_resampled, y_resampled = handle_imbalanced_data(X, y, method='smote')

    # Train-test split using StratifiedKFold for better handling of imbalance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    
    # Model selection and hyperparameter tuning
    model, model_name = model_selection(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

      # Ensure you work with a copy of eligible_players to avoid SettingWithCopyWarning
    eligible_players = eligible_players.copy()
    
    # Add 'match_frequency' for players based on the game timetable
    match_frequency = game_timetable_df[game_timetable_df['team_name'] == team_name]['player_name'].value_counts()
    eligible_players.loc[:, 'match_frequency'] = eligible_players['name'].map(match_frequency).fillna(0)

    # Add 'experience' (for example, number of matches played by each player)
    experience = game_timetable_df.groupby('player_name').size()
    eligible_players.loc[:, 'experience'] = eligible_players['name'].map(experience).fillna(0)
    
    eligible_players = add_new_features_eligible_players(eligible_players, game_timetable_df)
    
    
      # Fit model before predict proba
    model = model.fit(X_resampled,y_resampled)
     # Select the top N players based on the prediction (for example, top 11 players)
    eligible_players.loc[:, 'selection_probability'] = model.predict_proba(
        eligible_players[['power', 'match_frequency', 'experience','defensive_skill', 'offensive_skill','avg_goals', 'avg_assists', 'coach_rating']]
    )[:, 1]
    selected_players = eligible_players.sort_values(by='selection_probability', ascending=False).head(11)

      # --- Plotting code ---
    sns.boxplot(x=eligible_players['selection_probability'])
    plt.title("Selection Probability Distribution")
    plt.show()

    sns.scatterplot(data=eligible_players, x='avg_goals', y='avg_assists', hue='selection_probability')
    plt.title("Goals vs Assists with Selection Probability")
    plt.show()
     # --- End Plotting code ---
    
    plan = select_plan_with_ml(selected_players, game_history_df, team_name)
    player_positions = distribute_players_with_ml(selected_players, plan, player_data_df)
    return selected_players, plan, player_positions

def add_new_features(player_data_df, game_timetable_df):
    # Calculate average goals and assists (example)
    goals = game_timetable_df.groupby('player_name')['goals'].mean().fillna(0)
    assists = game_timetable_df.groupby('player_name')['assists'].mean().fillna(0)
    player_data_df.loc[:, 'avg_goals'] = player_data_df['name'].map(goals).fillna(0).astype(float)
    player_data_df.loc[:, 'avg_assists'] = player_data_df['name'].map(assists).fillna(0).astype(float)

        # Calculate average minutes played
    minutes_played = game_timetable_df.groupby('player_name')['minutes_played'].mean().fillna(0)
    player_data_df.loc[:, 'avg_minutes_played'] = player_data_df['name'].map(minutes_played).fillna(0).astype(float)

    # Calculate yellow card rate and red card rate
    yellow_cards = game_timetable_df.groupby('player_name')['yellow_cards'].sum().fillna(0)
    red_cards = game_timetable_df.groupby('player_name')['red_cards'].sum().fillna(0)
    total_matches = game_timetable_df['player_name'].value_counts().fillna(0)

    player_data_df.loc[:, 'yellow_card_rate'] = (player_data_df['name'].map(yellow_cards) / player_data_df['name'].map(total_matches)).fillna(0).astype(float)
    player_data_df.loc[:, 'red_card_rate'] = (player_data_df['name'].map(red_cards) / player_data_df['name'].map(total_matches)).fillna(0).astype(float)

    # Add a dummy coach rating (you would replace this with real data)
    player_data_df.loc[:, 'coach_rating'] = player_data_df['name'].apply(lambda x: np.random.randint(1,5) if x not in ['John Smith','Alice Brown', 'Charlie Green'] else 5 )  # Example dummy rating

      #  Add total performace score
    player_data_df.loc[:, 'total_performance_score'] = player_data_df[['power', 'speed', 'stamina', 'agility']].sum(axis=1).astype(float)

     # Add a ratio of offensive vs stamina skill
    player_data_df.loc[:, 'offensive_stamina_ratio'] = (player_data_df['offensive_skill'] / player_data_df['stamina']).astype(float)

    return player_data_df

def add_new_features_eligible_players(eligible_players, game_timetable_df):
    # Calculate average goals and assists (example)
    goals = game_timetable_df.groupby('player_name')['goals'].mean().fillna(0)
    assists = game_timetable_df.groupby('player_name')['assists'].mean().fillna(0)
    eligible_players.loc[:, 'avg_goals'] = eligible_players['name'].map(goals).fillna(0).astype(float)
    eligible_players.loc[:, 'avg_assists'] = eligible_players['name'].map(assists).fillna(0).astype(float)

    # Calculate average minutes played
    minutes_played = game_timetable_df.groupby('player_name')['minutes_played'].mean().fillna(0)
    eligible_players.loc[:, 'avg_minutes_played'] = eligible_players['name'].map(minutes_played).fillna(0).astype(float)
      # Calculate yellow card rate and red card rate
    yellow_cards = game_timetable_df.groupby('player_name')['yellow_cards'].sum().fillna(0)
    red_cards = game_timetable_df.groupby('player_name')['red_cards'].sum().fillna(0)
    total_matches = game_timetable_df['player_name'].value_counts().fillna(0)

    eligible_players.loc[:, 'yellow_card_rate'] = (eligible_players['name'].map(yellow_cards) / eligible_players['name'].map(total_matches)).fillna(0).astype(float)
    eligible_players.loc[:, 'red_card_rate'] = (eligible_players['name'].map(red_cards) / eligible_players['name'].map(total_matches)).fillna(0).astype(float)

    # Add a dummy coach rating (you would replace this with real data)
    eligible_players.loc[:, 'coach_rating'] = eligible_players['name'].apply(lambda x: np.random.randint(1,5) if x not in ['John Smith','Alice Brown', 'Charlie Green'] else 5 )  # Example dummy rating

    #  Add total performace score
    eligible_players.loc[:, 'total_performance_score'] = eligible_players[['power', 'speed', 'stamina', 'agility']].sum(axis=1).astype(float)

     # Add a ratio of offensive vs stamina skill
    eligible_players.loc[:, 'offensive_stamina_ratio'] = (eligible_players['offensive_skill'] / eligible_players['stamina']).astype(float)
    
    return eligible_players

def handle_imbalanced_data(X, y, method='smote'):
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        undersample = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersample.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X.copy(), y.copy()
    return X_resampled, y_resampled

def model_selection(X_train, y_train):
    # Parameter grids for hyperparameter tuning
    param_grid_rf = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [5, 10, 15],
        'model__min_samples_leaf': [2, 5, 10]
    }

    param_grid_gb = {
       'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
    }
    param_grid_xgb = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
        'model__subsample': [0.8, 1],
    'model__colsample_bytree': [0.8, 1]
}
    # Models
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), param_grid_rf),
       "GradientBoosting": (GradientBoostingClassifier(random_state=42), param_grid_gb),
       "XGBoost":(XGBClassifierWrapper(random_state=42),param_grid_xgb)
    }

    best_model = None
    best_score = 0
    best_name = ""
    # Grid Search for all models
    for name, (model,param_grid) in models.items():
        pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize data before PCA
        ('pca', PCA(n_components=0.95)), # use PCA
        ('model', model) # Model
    ])
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_name = name

    return best_model,best_name

def select_opponent_players(opponent_team, match_date, selected_players):
    # Read relevant CSV files
    team_df = pd.read_csv('team.csv')
    game_timetable_df = pd.read_csv('game-timetable.csv')

    # Filter for the opponent team and match date
    opponent_players = team_df[team_df['team_name'] == opponent_team].copy()  # Use .copy() to avoid SettingWithCopyWarning
    match_details = game_timetable_df[
        (game_timetable_df['team_name'] == opponent_team) & 
        (game_timetable_df['match_date'] == match_date)
    ]
    
    if match_details.empty:
        raise ValueError("Match details for opponent team not found. Please check the input data.")
    
    # Rule 1: Anti-action logic
    max_power = selected_players['power'].max()
    selected_opponent_players = opponent_players[
        opponent_players['anti_action'] >= max_power
    ].copy()  # Use .copy() for safety

    if selected_opponent_players.empty:
        raise ValueError("No opponent players meet the anti-action criteria.")
    
    # Rule 2: Frequent matches against the team
    opponent_team_name = selected_players['team_name'].iloc[0]
    match_frequency = game_timetable_df[
        (game_timetable_df['team_name'] == opponent_team) & 
        (game_timetable_df['opponent'] == opponent_team_name)
    ]['player_name'].value_counts()

    # Add match frequency as a column
    selected_opponent_players['match_frequency'] = selected_opponent_players['name'].map(match_frequency).fillna(0)

    # Sort players by match frequency (descending)
    selected_opponent_players = selected_opponent_players.sort_values(by='match_frequency', ascending=False)

    return selected_opponent_players

def select_plan(selected_players, game_history_df, team_name):
    """Selects a plan based on selected player characteristics and historical match results."""
    avg_offensive_skill = selected_players['offensive_skill'].mean()
    avg_defensive_skill = selected_players['defensive_skill'].mean()
    avg_speed = selected_players['speed'].mean()
    avg_stamina = selected_players['stamina'].mean()

    offensive_focus = "Medium"
    if avg_offensive_skill > 80:
        offensive_focus = "High"
    elif avg_offensive_skill < 70:
      offensive_focus = "Low"

    defensive_style = "Balanced"
    if avg_defensive_skill > 80:
        defensive_style = "Aggressive"
    elif avg_defensive_skill < 70:
        defensive_style = "Passive"
        
    pace = "Normal"
    if avg_speed > 80:
        pace = "Fast"
    elif avg_stamina < 70:
        pace = "Slow"
    
    formations = ["4-3-3", "4-4-2", "3-5-2"]
    formation = random.choice(formations)
    # Check historical win rate with the same plan
    team_history = game_history_df[game_history_df['team_name'] == team_name]
    team_history = team_history[
      (team_history['plan_offensive_focus'] == offensive_focus) &
        (team_history['plan_defensive_style'] == defensive_style) &
        (team_history['plan_pace'] == pace)
    ]
    wins = (team_history['winning_team'] == team_name).sum()
    total = len(team_history)

    win_rate = wins/total if total > 0 else 0
    if win_rate < 0.3:
       if avg_offensive_skill > 80:
            offensive_focus = "Medium"
       elif avg_offensive_skill < 70:
            offensive_focus = "Medium"
       if avg_defensive_skill > 80:
          defensive_style = "Balanced"
       elif avg_defensive_skill < 70:
          defensive_style = "Balanced"
       if avg_speed > 80:
         pace = "Normal"
       elif avg_stamina < 70:
           pace = "Normal"
    
    return {
        "offensive_focus": offensive_focus,
        "defensive_style": defensive_style,
        "pace": pace,
        "formation": formation
    }

def select_plan_with_ml(selected_players, game_history_df, team_name):
    """Selects a plan based on selected player characteristics and historical match results using ML."""
    # Prepare data for plan prediction
    plan_features = [
        'offensive_skill',
        'defensive_skill',
        'speed',
        'stamina',
    ]

    
    player_data_avg = selected_players[plan_features].mean().to_dict()
    
    historical_plan_data = game_history_df[
        ['plan_offensive_focus','plan_defensive_style','plan_pace', 'selected_players']
    ].copy()

    historical_plan_data.rename(columns={
      'plan_offensive_focus': 'offensive_focus',
      'plan_defensive_style': 'defensive_style',
      'plan_pace': 'pace'}, inplace = True)
    
    historical_plan_data['formation'] = game_history_df['selected_players'].apply(lambda players: random.choice(["4-3-3", "4-4-2", "3-5-2"]))
    
    
    
    y_plan = historical_plan_data.copy()
    if len(y_plan) < 10:
         return select_plan(selected_players, game_history_df, team_name)
    
    y_plan['plan'] = y_plan[['offensive_focus', 'defensive_style', 'pace', 'formation']].agg('-'.join, axis=1)
   
    X_plan = pd.DataFrame([player_data_avg], columns=plan_features)
    
    X_plan_dummies = pd.get_dummies(historical_plan_data[["offensive_focus","defensive_style","pace", "formation"]])
    
    
    X_plan = pd.concat([X_plan, X_plan_dummies.reindex(columns=X_plan_dummies.columns, fill_value=0)], axis = 1).fillna(0)
    
    
    # Train a model to predict plan
    X_train_plan, X_test_plan, y_train_plan, y_test_plan = train_test_split(X_plan, y_plan['plan'], test_size=0.2, random_state=42)

    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [2, 5, 10]
    }
    model_plan = RandomForestClassifier(random_state=42)
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=3, shuffle=True, random_state = 42)
    grid_search = GridSearchCV(estimator=model_plan, param_grid=param_grid_rf, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_plan, y_train_plan)
    best_model_plan = grid_search.best_estimator_
    
    # Use trained model to predict the plan
    
    predicted_plan = best_model_plan.predict(X_plan.iloc[[0]])[0]
    
    if isinstance(predicted_plan,str):
      try:
        offensive_focus, defensive_style, pace, formation  = predicted_plan.split('-')
      except ValueError:
        return select_plan(selected_players, game_history_df, team_name)
    else:
         return select_plan(selected_players, game_history_df, team_name)
    return {
        "offensive_focus": offensive_focus,
        "defensive_style": defensive_style,
        "pace": pace,
        "formation" : formation
    }

def distribute_players_with_ml(selected_players, plan, player_data_df):
    """Distributes players to positions based on ML model"""

    available_positions = ["GK", "LB", "RB", "CB", "CB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "ST"]

    positions = {}

    eligible_players = selected_players.copy()

    player_positions_df = player_data_df.copy()

    if len(player_positions_df) < 10:
       positions = distribute_players(eligible_players, plan["formation"])
       return positions

    X_positions = player_positions_df[
      ['power', 'speed', 'stamina', 'agility', 'defensive_skill', 'offensive_skill']
      ].copy()

    # Debugging: Print positions before filtering
    # print("Positions before filtering:", player_positions_df['position'])

    player_positions_df['position'] = player_positions_df['position'].apply(lambda x: x if x in list(POSITIONS.values()) else "Unknown")

    # Debugging: Print positions after marking unknowns
    #print("Positions after marking unknowns:", player_positions_df['position'])
    
    # Check if all positions were filtered out
    if player_positions_df['position'].isin(['Unknown']).all():
         print("All positions were filtered, using a default approach.")
         return distribute_players(eligible_players, plan["formation"])


    player_positions_df = player_positions_df[player_positions_df['position'] != "Unknown"]

    # Debugging: Print positions after removing unknowns
    #print("Positions after filtering unknowns:", player_positions_df['position'])


    y_positions = player_positions_df['position'].apply(lambda x: list(POSITIONS.keys())[list(POSITIONS.values()).index(x)])
    
    #Debug: print number of rows for X and Y positions
    # print("Shape of X_positions:", X_positions.shape)
    # print("Shape of y_positions:", y_positions.shape)


    # Check if y_positions is empty now
    if y_positions.empty:
      print("No valid positions found, using default player distribution")
      return distribute_players(eligible_players, plan["formation"])

    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X_positions, y_positions, test_size=0.2, random_state=42)

    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [2, 5, 10]
    }
    model_position = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model_position, param_grid=param_grid_rf, cv=3, scoring='accuracy')
    grid_search.fit(X_train_pos, y_train_pos)
    best_model_position = grid_search.best_estimator_

    #Assign the players to the positions.
    for player in selected_players['name']:
        player_features = player_data_df[player_data_df['name']==player][['power', 'speed', 'stamina', 'agility', 'defensive_skill', 'offensive_skill']].copy()
        if not player_features.empty:
            predicted_position = best_model_position.predict(player_features)[0]
            positions[player]= POSITIONS[predicted_position]
        else:
          positions[player] = "unknown"
    return positions

def create_match_timetable(selected_players, selected_opponent_players, match_date, plan, player_positions):
    match_timetable = []

    # Example logic for player entry
    for player in selected_players['name']:
        match_timetable.append({
            'player_name': player,
            'team': selected_players['team_name'].iloc[0],
            'entry_time': 'start',
            'match_date': match_date,
            'plan' : plan,
            'position': player_positions[player]
        })

    for player in selected_opponent_players['name']:
        match_timetable.append({
            'player_name': player,
            'team': selected_opponent_players['team_name'].iloc[0],
            'entry_time': 'start',
            'match_date': match_date,
            'plan': plan,
             'position': 'unknown'
        })

    match_timetable_df = pd.DataFrame(match_timetable)
    match_timetable_df.to_csv('match_timetable.csv', index=False)
    return match_timetable_df
def distribute_players(selected_players, formation):
    """Distributes players to positions based on the formation, not using ML"""
    positions = {}
    available_players = selected_players['name'].tolist()
    if formation == "4-3-3":
        position_list = ["GK", "LB", "RB", "CB", "CB", "CDM", "CM", "CAM", "LW", "RW", "ST"]
    elif formation == "4-4-2":
        position_list = ["GK", "LB", "RB", "CB", "CB", "LM", "RM", "CM", "CM", "ST", "ST"]
    elif formation == "3-5-2":
        position_list = ["GK", "CB", "CB", "CB", "LM", "RM", "CM", "CM", "CAM", "ST", "ST"]
    else:
        return {}

    for position in position_list:
         if available_players:
            player = random.choice(available_players)
            positions[player]= POSITIONS[position]
            available_players.remove(player)
    return positions

def generate_result_txt(selected_players, plan, player_positions):
    with open('result.txt', 'w') as f:
        f.write("Selected Players:\n")
        for index, player in selected_players.iterrows():
            f.write(f"{player['name']} - Position: {player_positions[player['name']]} - Selection Probability: {player['selection_probability']}\n")
        f.write("\nSelected Plan:\n")
        for key, value in plan.items():
            f.write(f"{key}: {value}\n")


team_name = 'Team A'
match_date = '2024-01-10'

# Load necessary data
game_timetable_df = pd.read_csv('game-timetable.csv')
team_df = pd.read_csv('team.csv')
player_data_df = pd.read_csv('player_data.csv')
game_history_df = pd.read_csv('game_history.csv')
# Get eligible players
eligible_players = team_df[team_df['team_name'] == team_name]

# Call the function with correct arguments
selected_players, plan, player_positions = select_players(team_name, match_date)
# Print the selected players
print(selected_players)
print ("Selected Plan:", plan)
print ("Players in their positions:", player_positions)
generate_result_txt(selected_players, plan, player_positions)




def visualize_match_and_save(selected_players, player_positions):
    """Visualizes player positions on a football field and saves the plot as a PNG file."""
    plt.figure(figsize=(10, 7))
    filename="result.png"
    # Draw the field
    plt.gca().add_patch(plt.Rectangle((0, 0), 100, 80, fill=None, edgecolor="black", lw=2))
    plt.plot([50, 50], [0, 80], color="black", lw=2)
    plt.plot([10,10],[20,60], color="black", lw=2)
    plt.plot([90,90],[20,60], color="black", lw=2)
    plt.plot([0, 0], [0, 0], color="black", lw=2) #to show the plot limits

    # Define Positions
    positions = {
        "Goalkeeper": (50, 5),
        "Left Back": (15, 20),
        "Right Back": (85, 20),
        "Center Back": [(30, 40),(70,40), (50, 20)],
        "Central Defensive Midfielder": (50, 50),
        "Central Midfielder": [(40,60),(60,60)],
        "Central Attacking Midfielder": (50, 70),
        "Left Midfielder": (20, 50),
        "Right Midfielder": (80, 50),
        "Left Winger": (15, 70),
        "Right Winger": (85, 70),
        "Striker": [(35,75), (65,75), (50, 75)]
    }

    def calculate_pos(position, index=None):
        if isinstance(positions[position], list):
            if index is None:
                return positions[position][0]
            else:
                return positions[position][index % len(positions[position])]
        else:
            return positions[position]

    # Colors
    cmap = plt.cm.Greens
    max_prob = selected_players['selection_probability'].max()
    min_prob = selected_players['selection_probability'].min()

    # Calculate relative size and color intensity based on probability
    for index, player in selected_players.iterrows():
        if player_positions.get(player["name"]) in positions.keys():
            x_pos, y_pos = calculate_pos(player_positions.get(player["name"]), index)
            size_value = 100 + (player['selection_probability'] - min_prob) * 300
            color_value = cmap((player['selection_probability'] - min_prob) / (max_prob - min_prob) if max_prob != min_prob else 0)

            plt.scatter(x_pos, y_pos, s=size_value, color=color_value, edgecolor="black")
            plt.text(x_pos - 3, y_pos + 2, player['name'], ha="center", fontsize=9)
        else:
            print(f"Position {player_positions.get(player['name'])} not found for player {player['name']}")

    plt.xlim(-1, 101)
    plt.ylim(-1, 81)
    plt.title("Player Positions and Selection Probability")
    plt.xlabel("Field Width")
    plt.ylabel("Field Height")
    plt.xticks([])
    plt.yticks([])

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()


visualize_match_and_save(selected_players, player_positions)



