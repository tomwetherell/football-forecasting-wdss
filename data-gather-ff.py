""" 
Football Forecasting Competition
Functions for gathering, tidying and manipulating the data
"""

# Standard imports
import numpy as np
import pandas as pd

# Data processing
from sklearn.preprocessing import OrdinalEncoder


def get_premier_league_data(start_year: int) -> pd.core.frame.DataFrame:
    """
    A function to retrieve, clean and tidy premier league data
    """
    # Retrieve the data
    season = str(start_year)[-2:] + str(start_year + 1)[-2:]
    data = pd.read_csv("data/prem-data-" + season + ".csv")

    if start_year > 2018:
        # Filtering columns of interest
        columns = [
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "AvgH",
            "AvgD",
            "AvgA",
            "Date",
        ]
        data = data[columns]

        # Renaming columns
        data = data.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})

    if start_year <= 2018:
        # Filtering columns of interest
        columns = [
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "B365H",
            "B365D",
            "B365A",
            "Date",
        ]
        data = data[columns]

        # Renaming columns
        data = data.rename(
            columns={
                "FTHG": "HomeGoals",
                "FTAG": "AwayGoals",
                "B365H": "AvgH",
                "B365D": "AvgD",
                "B365A": "AvgA",
            }
        )

    # Remove final week of fixtures
    data = data[:-10]

    # Remove games involving teams that are not currently in the premier league
    non_prem_teams = [
        "Sheffield United",
        "Fulham",
        "West Brom",
        "Bournemouth",
        "Swansea",
        "Stoke",
        "Cardiff",
        "Huddersfield",
    ]
    for team in non_prem_teams:
        data.drop(data.loc[data["HomeTeam"] == team].index, inplace=True)
        data.drop(data.loc[data["AwayTeam"] == team].index, inplace=True)

    # Separate home goals data
    home_goals = data[
        ["HomeTeam", "AwayTeam", "HomeGoals", "AvgH", "AvgD", "AvgA", "Date"]
    ]
    home_goals = home_goals.assign(home=1)
    home_goals = home_goals.rename(
        columns={
            "HomeTeam": "team",
            "AwayTeam": "opponent",
            "Date": "date",
            "AvgH": "team_odds",
            "AvgD": "draw_odds",
            "AvgA": "oppo_odds",
            "HomeGoals": "goals",
        }
    )

    # Separate away goals data
    away_goals = data[
        ["HomeTeam", "AwayTeam", "AwayGoals", "AvgH", "AvgD", "AvgA", "Date"]
    ]
    away_goals = away_goals.assign(home=0)
    away_goals = away_goals.rename(
        columns={
            "HomeTeam": "opponent",
            "AwayTeam": "team",
            "Date": "date",
            "AvgH": "oppo_odds",
            "AvgD": "draw_odds",
            "AvgA": "team_odds",
            "AwayGoals": "goals",
        }
    )

    # Concatenating into training data
    training_data = pd.concat([home_goals, away_goals])  # 740 rows
    training_data = training_data.reset_index(drop=True)

    return training_data


def get_fantasy_data(start_year: int) -> pd.core.frame.DataFrame:
    """Retrieves and manipulates fantasy football data"""
    # Retrieve the data
    season = str(start_year) + "-" + str(start_year + 1)[-2:]
    data = pd.read_csv(
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
        + season
        + "/teams.csv"
    )

    # Filtering columns of interest in the fantasy data
    columns = [
        "name",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    data = data[columns]

    # Remove games involving teams that are not currently in the premier league
    non_prem_teams = [
        "Sheffield United",
        "Fulham",
        "West Brom",
        "Bournemouth",
        "Swansea",
        "Stoke",
        "Cardiff",
        "Huddersfield",
    ]
    for team in non_prem_teams:
        data.drop(data.loc[data["name"] == team].index, inplace=True)

    # Renaming for constistency
    data.loc[data["name"] == "Spurs", "name"] = "Tottenham"
    data.loc[data["name"] == "Man Utd", "name"] = "Man United"
    data.loc[data["name"] == "Spurs", "name"] = "Tottenham"
    data.loc[data["name"] == "Man Utd", "name"] = "Man United"

    # Averaging home/away columns (may change later - this makes it easier for now)
    str_atk_avg = (
        data.loc[:, "strength_attack_home"] + data.loc[:, "strength_attack_away"]
    ) / 2
    data.loc[:, "fantasy_strength_atk"] = str_atk_avg
    str_def_avg = (
        data.loc[:, "strength_defence_home"] + data.loc[:, "strength_defence_away"]
    ) / 2
    data.loc[:, "fantasy_strength_def"] = str_def_avg
    data.drop(
        [
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
        ],
        axis=1,
        inplace=True,
    )

    data["name"] = data["name"].astype("str")

    data.reset_index(inplace=True, drop=True)

    return data


def combine_prem_fan_data(
    prem_data: pd.core.frame.DataFrame, fan_data: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """
    Combines premier league data (loaded using get_premier_league_data) with
    fantasy football data (loaded using get_fantasy_data) from the same year
    """
    team_fantasy_atk_scores = [
        fan_data.fantasy_strength_atk[np.where(fan_data["name"] == team_name)[0][0]]
        for team_name in prem_data.team
    ]
    prem_data["team_fantasy_atk_str"] = team_fantasy_atk_scores

    team_fantasy_def_scores = [
        fan_data.fantasy_strength_def[np.where(fan_data["name"] == team_name)[0][0]]
        for team_name in prem_data.team
    ]
    prem_data["team_fantasy_def_str"] = team_fantasy_def_scores

    oppo_fantasy_atk_scores = [
        fan_data.fantasy_strength_atk[np.where(fan_data["name"] == team_name)[0][0]]
        for team_name in prem_data.opponent
    ]
    prem_data["oppo_fantasy_atk_str"] = oppo_fantasy_atk_scores

    oppo_fantasy_def_scores = [
        fan_data.fantasy_strength_def[np.where(fan_data["name"] == team_name)[0][0]]
        for team_name in prem_data.opponent
    ]
    prem_data["oppo_fantasy_def_str"] = oppo_fantasy_def_scores

    return prem_data


def team_goals_recent(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Adding team_goals_last5 feature, which denotes the total number of goals the team has scored in their last 5 games.
    Captures whether a team is on a streak, or in a rough patch.
    """
    prem_teams = [
        "Man City",
        "Man United",
        "Liverpool",
        "Chelsea",
        "Leicester",
        "West Ham",
        "Tottenham",
        "Arsenal",
        "Leeds",
        "Everton",
        "Aston Villa",
        "Newcastle",
        "Wolves",
        "Crystal Palace",
        "Southampton",
        "Brighton",
        "Burnley",
        "Norwich",
        "Watford",
        "Brentford",
    ]

    team_dataframes_list = []

    for team in prem_teams:
        team_df = dataset.loc[dataset.team == team, :]
        team_df = team_df.reset_index(drop=True)
        team_df["date"] = pd.to_datetime(team_df["date"], dayfirst=True)
        team_df = team_df.sort_values(by="date")
        team_df = team_df.reset_index(drop=True)
        avg_num_goals_for_team = team_df.goals.mean()
        last5_goals = np.zeros(len(team_df))
        for idx in range(len(team_df)):
            # For the first 5 games of a team each season, we can't consider the sum of the
            # their last 5 games. To substitute, we consider their average num goals * 5
            if idx in [0, 1, 2, 3, 4]:
                last5_goals[idx] = avg_num_goals_for_team * 5
            else:
                last5_goals[idx] = (
                    team_df.goals[idx - 1]
                    + team_df.goals[idx - 2]
                    + team_df.goals[idx - 3]
                    + team_df.goals[idx - 4]
                    + team_df.goals[idx - 5]
                )
        team_df.loc[:, "team_goals_last5"] = last5_goals
        team_dataframes_list.append(team_df)

    combined_dataset = pd.concat(team_dataframes_list)
    combined_dataset["date"] = pd.to_datetime(combined_dataset["date"], dayfirst=True)

    return_dataset = combined_dataset.sort_values(by="date")
    return_dataset = return_dataset.reset_index(drop=True)

    return return_dataset


def oppo_goals_recent(dataset: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Adds oppo_goals_last5 feature"""

    prem_teams = [
        "Man City",
        "Man United",
        "Liverpool",
        "Chelsea",
        "Leicester",
        "West Ham",
        "Tottenham",
        "Arsenal",
        "Leeds",
        "Everton",
        "Aston Villa",
        "Newcastle",
        "Wolves",
        "Crystal Palace",
        "Southampton",
        "Brighton",
        "Burnley",
        "Norwich",
        "Watford",
        "Brentford",
    ]

    team_dataframes_list = []

    for team in prem_teams:
        team_df = dataset.loc[dataset.team == team, :]
        team_df = team_df.reset_index(drop=True)
        team_df["date"] = pd.to_datetime(team_df["date"], dayfirst=True)
        team_df = team_df.sort_values(by="date")
        team_df = team_df.reset_index(drop=True)
        oppo_df = dataset.loc[dataset.opponent == team, :]
        oppo_df = oppo_df.reset_index(drop=True)
        oppo_df["date"] = pd.to_datetime(oppo_df["date"], dayfirst=True)
        oppo_df = oppo_df.sort_values(by="date")
        oppo_df = oppo_df.reset_index(drop=True)
        team_df.loc[:, "oppo_goals_last5"] = oppo_df["team_goals_last5"]
        team_dataframes_list.append(team_df)

    combined_dataset = pd.concat(team_dataframes_list)
    combined_dataset["date"] = pd.to_datetime(combined_dataset["date"], dayfirst=True)
    return_dataset = combined_dataset.sort_values(by="date")
    return_dataset = return_dataset.reset_index(drop=True)

    return return_dataset


def ordinal_encode_teamoppo(
    dataset: pd.core.frame.DataFrame,
) -> pd.core.frame.DataFrame:
    """Ordinal encode the 'team' and 'opponent' categorical columns"""

    prem_teams = [
        "Man City",
        "Liverpool",
        "Chelsea",
        "Man United",
        "West Ham",
        "Arsenal",
        "Wolves",
        "Tottenham",
        "Brighton",
        "Southampton",
        "Leicester",
        "Aston Villa",
        "Crystal Palace",
        "Brentford",
        "Leeds",
        "Everton",
        "Newcastle",
        "Norwich",
        "Watford",
        "Burnley",
    ]

    ordinal_encoder = OrdinalEncoder(categories=[prem_teams])
    dataset.team = ordinal_encoder.fit_transform(dataset.loc[:, ["team"]])

    ordinal_encoder2 = OrdinalEncoder(categories=[prem_teams])
    dataset.opponent = ordinal_encoder2.fit_transform(dataset.loc[:, ["opponent"]])

    # Ensuring entries are sorted by date
    return_dataset = dataset.sort_values(by="date")

    return return_dataset


def make_dataset(years: list) -> pd.core.frame.DataFrame:
    """
    Makes the final dataset for use in training
    """
    # List of datasets. Each element is a seperate year's dataset
    prem_data = [get_premier_league_data(year) for year in years]
    fantasy_data = [get_fantasy_data(year) for year in years]
    combined_data = [
        combine_prem_fan_data(prem_data[i], fantasy_data[i]) for i in range(len(years))
    ]

    combined_dataset = pd.concat(combined_data)
    combined_dataset.reset_index(drop=True, inplace=True)

    # Adding team_goals_last5 feature
    dataset_teamgoals = team_goals_recent(combined_dataset)
    # Adding oppo_goals_last5 feature
    dataset_oppogoals = oppo_goals_recent(dataset_teamgoals)

    # Ordinal encoding the 'team' and 'opponenet' categorical columns
    encoded_dataset = ordinal_encode_teamoppo(dataset_oppogoals)

    return encoded_dataset
