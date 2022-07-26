''' 
Football Forecasting Competition
Functions for gathering, tidying and manipulating the data
'''

# Standard imports
import numpy as np 
import pandas as pd 


def get_premier_league_data(start_year: int) -> pd.core.frame.DataFrame:
    '''
    A function to retrieve, clean and tidy premier league data
    '''
    # Retrieve the data
    season = str(start_year)[-2:] + str(start_year + 1)[-2:]
    data = pd.read_csv("data/prem-data-" + season + ".csv") 
    
    if start_year > 2018:
        # Filtering columns of interest
        columns = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "AvgH", "AvgD", "AvgA", "Date"]
        data = data[columns]

        # Renaming columns 
        data = data.rename(columns = {
                         "FTHG": "HomeGoals",
                         "FTAG": "AwayGoals"
                         }
        )
    
    if start_year <= 2018:
        # Filtering columns of interest
        columns = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A", "Date"]
        data = data[columns]

        # Renaming columns 
        data = data.rename(columns = {
                         "FTHG": "HomeGoals",
                         "FTAG": "AwayGoals",
                         "B365H": "AvgH",
                         "B365D": "AvgD",
                         "B365A": "AvgA",
                         })
    
    # Remove final week of fixtures
    data = data[:-10]
    
    # Remove games involving teams that are not currently in the premier league
    non_prem_teams = ['Sheffield United', 'Fulham', 'West Brom', 'Bournemouth',
                     'Swansea', 'Stoke', 'Cardiff', 'Huddersfield']
    for team in non_prem_teams:
        data.drop(data.loc[data['HomeTeam']==team].index, inplace=True)
        data.drop(data.loc[data['AwayTeam']==team].index, inplace=True)
        
    # Separate home goals data
    home_goals = data[["HomeTeam", "AwayTeam", "HomeGoals", "AvgH", "AvgD", "AvgA", "Date"]]
    home_goals = home_goals.assign(home=1)
    home_goals = home_goals.rename(
        columns={"HomeTeam": "team",
                 "AwayTeam": "opponent",
                 "Date": "date",
                 "AvgH": "team_odds",
                 "AvgD": "draw_odds",
                 "AvgA": "oppo_odds",
                 "HomeGoals": "goals"}
    )

    # Separate away goals data 
    away_goals = data[["HomeTeam", "AwayTeam", "AwayGoals", "AvgH", "AvgD", "AvgA", "Date"]]
    away_goals = away_goals.assign(home=0)
    away_goals = away_goals.rename(
        columns={"HomeTeam": "opponent",
                 "AwayTeam": "team", 
                 "Date": "date",
                 "AvgH": "oppo_odds",
                 "AvgD": "draw_odds",
                 "AvgA": "team_odds",
                 "AwayGoals": "goals"}
    )
    
    # Concatenating into training data 
    training_data = pd.concat([home_goals, away_goals]) # 740 rows 
    training_data = training_data.reset_index(drop=True)
 
    return training_data


def get_fantasy_data(start_year: int) -> pd.core.frame.DataFrame:
    ''' Retrieves and manipulates fantasy football data'''
    # Retrieve the data
    season = str(start_year) + '-' + str(start_year+1)[-2:]
    data = pd.read_csv("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/" + season + "/teams.csv")
    
    # Filtering columns of interest in the fantasy data
    columns = ["name", "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]
    data = data[columns]
    
    # Remove games involving teams that are not currently in the premier league
    non_prem_teams = ['Sheffield United', 'Fulham', 'West Brom', 'Bournemouth',
                     'Swansea', 'Stoke', 'Cardiff', 'Huddersfield']
    for team in non_prem_teams:
        data.drop(data.loc[data['name']==team].index, inplace=True)
        
    # Renaming for constistency    
    data.loc[data['name']=='Spurs', 'name'] = 'Tottenham'
    data.loc[data['name']=='Man Utd', 'name'] = 'Man United'
    data.loc[data['name']=='Spurs', 'name'] = 'Tottenham'
    data.loc[data['name']=='Man Utd', 'name'] = 'Man United'
        
    # Averaging home/away columns (may change later - this makes it easier for now)
    str_atk_avg = (data.loc[:, 'strength_attack_home'] + data.loc[:, 'strength_attack_away']) / 2
    data.loc[:, 'fantasy_strength_atk'] = str_atk_avg
    str_def_avg = (data.loc[:, 'strength_defence_home'] + data.loc[:, 'strength_defence_away']) / 2
    data.loc[:, 'fantasy_strength_def'] = str_def_avg
    data.drop(['strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away'], axis=1, inplace=True)
    
    data['name'] = data['name'].astype('str') 
    
    data.reset_index(inplace=True, drop=True)
    
    return data



def combine_prem_fan_data(prem_data: pd.core.frame.DataFrame, fan_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''
    Combines premier league data (loaded using get_premier_league_data) with
    fantasy football data (loaded using get_fantasy_data) from the same year
     '''
    team_fantasy_atk_scores = [fan_data.fantasy_strength_atk[np.where(fan_data['name']==team_name)[0][0]] for team_name in prem_data.team]
    prem_data['team_fantasy_atk_str'] = team_fantasy_atk_scores
    
    team_fantasy_def_scores = [fan_data.fantasy_strength_def[np.where(fan_data['name']==team_name)[0][0]] for team_name in prem_data.team]
    prem_data['team_fantasy_def_str'] = team_fantasy_def_scores
    
    oppo_fantasy_atk_scores = [fan_data.fantasy_strength_atk[np.where(fan_data['name']==team_name)[0][0]] for team_name in prem_data.opponent]
    prem_data['oppo_fantasy_atk_str'] = oppo_fantasy_atk_scores
    
    oppo_fantasy_def_scores = [fan_data.fantasy_strength_def[np.where(fan_data['name']==team_name)[0][0]] for team_name in prem_data.opponent]
    prem_data['oppo_fantasy_def_str'] = oppo_fantasy_def_scores

    return prem_data


def make_dataset(years: list) -> pd.core.frame.DataFrame:
    '''
    
    '''
    # List of datasets. Each element is a seperate year's dataset 
    prem_data = [get_premier_league_data(year) for year in years]
    fantasy_data = [get_fantasy_data(year) for year in years]
    combined_data = [combine_prem_fan_data(prem_data[i], fantasy_data[i]) for i in range(len(years))]

    final_dataset = pd.concat(combined_data)
    final_dataset.reset_index(drop=True, inplace=True)

    return final_dataset


