## Herein lies the functions defined for the project thus far. Filter_prob_cats is a function for filtering the appropriate problem categories. 
## loadprocess_data is a function that calls filter_prob_cats, and cleans it

from pathlib import Path
import pandas as pd


def filter_prob_cats(cats, prob_cats):
    """Filter the problems' categories to only retain those we are interested in.

    Parameters
    ----------
    cats: list
        A list of string each indicating a category.

    Returns
    -------
    str
        Either one of the categories, if part of the list we are interested in.
        Otherwise, the 'Other' category.

    """

    cats = list(map(lambda x: x.strip("'"), cats))
    for cat in cats:
        if cat in prob_cats:
            return cat

    try:
        return cats[0]
    except IndexError:
        return 'other'

def loadprocess_data(ROOT_DIR, prob_cats):
    ROOT_DIR = Path(ROOT_DIR)

    # Load the users, problems, and problem attempts from file
    users = pd.read_csv(ROOT_DIR.joinpath('Data', 'dev_gomagic_users_export.csv'))
    prob_attempts = pd.read_csv(ROOT_DIR.joinpath('Data', 'dev_gomagic_prob_attempts.csv'), index_col='ID')
    probs = pd.read_csv(ROOT_DIR.joinpath('Data', 'dev_gomagic_problems_export.csv'),
                        converters={'cats': lambda x: x[1:-1].split(',') if x != 'NULL' else []})

    # Filter out users with no data
    users = users.dropna(how='any', subset=['streak', 'streak_max'])

    # Remove NaN columns from both user and problem attempts
    users = users.dropna(axis=1, how='all')
    prob_attempts = prob_attempts.dropna(axis=1, how='all')

    # Display some rudimentary statistics
    print(f'Total number of users: {len(users)}')
    print(f'Total number of problems: {len(probs)}')

    print(f'Users that attempted problems: {prob_attempts["user_id"].nunique()}')
    print(f'Problems that were attempted: {prob_attempts["prob_id"].nunique()}')

    # Drop non-relevant columns from the problem attempts
    prob_attempts = prob_attempts.drop(labels=['quiz_attempt_id', 'time', 'hint_click', 'bug_report'], axis=1)

    # Rename the problem's ID and rank columns for ease of merging
    probs = probs.rename(columns={'ID': 'prob_id', 'rank': 'prob_rank'})

    # Drop all unnecessary columns from the problem table
    probs = probs.drop(labels=['where_is_used', 'tags'], axis=1)

    # Transform the category strings into categorical
    probs['cats'] = probs['cats'].apply(lambda x: filter_prob_cats(x, prob_cats)).astype('category')

    # Merge the problems with the attempts
    data = prob_attempts.merge(probs, on='prob_id')

    # Rename the user's ID and rank columns for ease of merging
    users = users.rename(columns={'ID': 'user_id', 'rank': 'user_rank'})

    # Merge the users with their attempts
    data = data.merge(users, on='user_id')

    # Replace the 'failed' (= 0) result with a -1 value to avoid mistaking it for a lack of connection between user and problem
    data.loc[data['result'] == 0, 'result'] = -1

    return data, users, prob_attempts, probs
