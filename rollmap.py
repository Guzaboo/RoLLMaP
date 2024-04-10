import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def getTeamStats(team):
    score = team['score']
    teamPlayerStats = []
    for player in team['players']:
        location = player['location']
        playerStats = [player['assists'], player['boost'], player['cartouches'], player['demos'], player['goals'],
                       location['x'], location['y'], location['z'], location['pitch'], location['roll'], location['yaw'],
                       player['saves'], player['score'], player['shots'], player['speed'], player['touches']]
        teamPlayerStats.append(playerStats)
    return torch.FloatTensor(score), torch.FloatTensor(teamPlayerStats)

for i in range(1, 39):
    f = open(f"..\\rocket-league-scribe\\recordings\\{i}.json", "r")

    doc = json.load(f)
    winner = doc['winner']
    history = doc['snapshots']
    for snapshot in history:
        time = torch.FloatTensor([snapshot['timeLeft'], snapshot['timePassed']])
        ball = snapshot['ball']
        ballStats = torch.FloatTensor([ball['speed'], ball['location']['x'], ball['location']['y'], ball['location']['z']])
        team1 = snapshot['team1']
        score1, team1Stats = getTeamStats(team1)
        team2 = snapshot['team2']
        score2, team2Stats = getTeamStats(team2)
        if snapshot == history[150]:
            print(team1)
            print(team2)
            print(ball)
            print()
            print(time)
            print(ballStats)
            print(score1, score2)
            print(team1Stats)
            print(team2Stats)

class RoLLMaPDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = []
        for d in data:
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)