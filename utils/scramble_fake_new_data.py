import pandas as pd
import os

def scramble_fake_new_data(old_data, state, scramble_perc):
    """ Generate testing dataset for great expectations.
    """
    data = pd.read_csv(old_data)
    data = data.sample(random_state = state, frac = scramble_perc)
    return data

old_data = os.getcwd() + "/data/" + [s for s in os.listdir('data/') if s.endswith(".csv")][0]

new_data = scramble_fake_new_data(old_data, 42, 0.3)


new_data.to_csv(os.getcwd() + "/new_data/new_data.csv")