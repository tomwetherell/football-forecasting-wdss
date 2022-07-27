""" 
Football Forecasting Competition
Functions for making predictions
"""

# Standard imports
import pandas as pd
import numpy as np

# For the model
from scipy.stats import poisson


def create_X(
    home_team,
    away_team,
    home_odds,
    draw_odds,
    away_odds,
    home_fan_atk_str,
    home_fan_def_str,
    away_fan_atk_str,
    away_fan_def_str,
    home_goals_last5,
    away_goals_last5,
):
    """
    Creates feature data frames for home and away team for the match
    """
    # Creating DataFrame for home team features
    X_home = pd.DataFrame(
        data={
            "team": home_team,
            "opponent": away_team,
            "team_odds": home_odds,
            "draw_odds": draw_odds,
            "oppo_odds": away_odds,
            "team_fantasy_atk_str": home_fan_atk_str,
            "team_fantasy_def_str": home_fan_def_str,
            "oppo_fantasy_atk_str": away_fan_atk_str,
            "oppo_fantasy_def_str": away_fan_def_str,
            "team_goals_last5": home_goals_last5,
            "oppo_goals_last5": away_goals_last5,
            "home": 1,
        },
        index=[1],
    )

    # Creating DataFrame for away team features
    X_away = pd.DataFrame(
        data={
            "team": away_team,
            "opponent": home_team,
            "team_odds": away_odds,
            "draw_odds": draw_odds,
            "oppo_odds": home_odds,
            "team_fantasy_atk_str": away_fan_atk_str,
            "team_fantasy_def_str": away_fan_def_str,
            "oppo_fantasy_atk_str": home_fan_atk_str,
            "oppo_fantasy_def_str": home_fan_def_str,
            "team_goals_last5": away_goals_last5,
            "oppo_goals_last5": home_goals_last5,
            "home": 0,
        },
        index=[1],
    )

    return X_home, X_away


def predict_avg_goals(X_home, X_away, model):
    """
    Predicts the expected number of goals for the home team and the away team.
    X_home: data frame containing the features for the home team. Generated by create_X
    X_away: data frame containing the features for the away team. Generated by create_X
    model: the trained model we are using for making predictions
    """
    # Predict the mean/expected number of goals for the home team
    home_goals_avg = model.predict(X_home)

    # Predict the mean/expected number of goals for the away team
    away_goals_avg = model.predict(X_away)

    return home_goals_avg, away_goals_avg


def predict_score_pmf(X_home, X_away, model, max_goals):
    """
    Returns the joint probability mass function for no. of goals scored by the home and away team.
    X_home: data frame containing the features for the home team. Generated by create_X
    X_away: data frame containing the features for the away team. Generated by create_X
    model: the trained model we are using for making predictions
    max_goals: the maximum number of goals
    """
    # Predict the average number of goals for home and away teams
    avg_goals = predict_avg_goals(X_home, X_away, model)
    home_goals_avg = avg_goals[0]
    away_goals_avg = avg_goals[1]

    # Compute marginal distribution for home goals
    home_goals_pmf = [poisson.pmf(i, home_goals_avg) for i in range(0, max_goals + 1)]

    # Compute marginal distribution for away goals
    away_goals_pmf = [poisson.pmf(i, away_goals_avg) for i in range(0, max_goals + 1)]

    # Compute joint distribution for match score as outer product. Can only use outer as we are assuming independence.v
    joint_pmf = np.outer(np.array(home_goals_pmf), np.array(away_goals_pmf))

    return joint_pmf


def predict_score(X_home, X_away, model):
    """
    Returns the score that the model predicts
    X_home: data frame containing the features for the home team. Generated by create_X
    X_away: data frame containing the features for the away team. Generated by create_X
    """
    # Predicting distribution of match scores
    score_pmf = predict_score_pmf(X_home, X_away, model, 10)
    score_pmf = score_pmf.round(3)
    # print(score_pmf)
    # print('')

    prob = np.max(score_pmf)

    # Computing distribution mode
    home_goals_mode = np.argmax(score_pmf) // (11)
    away_goals_mode = np.argmax(score_pmf) % (11)

    if home_goals_mode == 0 and away_goals_mode == 0:
        prob_star = (score_pmf[0, 0] + score_pmf[0, 1] + score_pmf[1, 0]) / 3
    elif home_goals_mode == 0:
        prob_star = (
            score_pmf[home_goals_mode, away_goals_mode]
            + score_pmf[home_goals_mode, away_goals_mode - 1]
            + score_pmf[home_goals_mode, away_goals_mode + 1]
            + score_pmf[home_goals_mode + 1, away_goals_mode]
        ) / 4
    elif away_goals_mode == 0:
        prob_star = (
            score_pmf[home_goals_mode, away_goals_mode]
            + score_pmf[home_goals_mode - 1, away_goals_mode]
            + score_pmf[home_goals_mode + 1, away_goals_mode]
            + score_pmf[home_goals_mode, away_goals_mode + 1]
        ) / 4
    else:
        prob_star = (
            score_pmf[home_goals_mode, away_goals_mode]
            + score_pmf[home_goals_mode + 1, away_goals_mode]
            + score_pmf[home_goals_mode - 1, away_goals_mode]
            + score_pmf[home_goals_mode, away_goals_mode + 1]
            + score_pmf[home_goals_mode, away_goals_mode - 1]
        ) / 5

    score_pred = (home_goals_mode, away_goals_mode)
    return score_pred, prob_star