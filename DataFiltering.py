import pandas as pd

df = pd.read_csv("league_data.csv")

# Something important to take in mind, column blueWins 1 represents blue team won,
# 0 means otehr team one (red team)

df["blueWins"].value_counts
blueWins = df["blueWins"].value_counts()
blueWins2 = blueWins.rename(index={0:'blue wins', 1:'red wins'})

print(blueWins2)

# the result is
# blue wins    4949
# red wins     4930

# to balance we will randomly remove 19 registries from victories so they are exactly 50-50

removed = df[df['blueWins'] == 0].sample(n=19)
df = df.drop(removed.index)

blueWins = df["blueWins"].value_counts()
blueWins2 = blueWins.rename(index={0:'blue wins', 1:'red wins'})

# now they should be 50-50
# blue wins    4930
# red wins     4930
print(blueWins2)

df_blue_wins =  df[df['blueWins'] == 0] 
df_red_wins = df[df['blueWins'] == 1]

# now we make the split for 80% training and 20% testing. Stargin by removing
blue_win_testing = df_blue_wins.sample(frac=0.2)
blue_win_training = df_blue_wins.drop(blue_win_testing.index)

red_win_testing = df_red_wins.sample(frac=0.2)
red_win_training = df_red_wins.drop(red_win_testing.index)

#Concatenating data farmes by training and tseting
training = pd.concat([blue_win_training, red_win_training])
testing = pd.concat([blue_win_testing, red_win_testing])

print("----- Training: \n", training["blueWins"].value_counts())
print("----- Testing: \n", testing["blueWins"].value_counts())

training.to_csv('lol_training.csv', index=False)
testing.to_csv('lol_testing.csv', index=False)

print("Training size: ", training.shape)
print("Testing size: ", testing.shape)