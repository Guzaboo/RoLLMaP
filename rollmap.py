import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

def getTeamStats(team):
    score = team['score']
    teamPlayerStats = []
    for player in team['players']:
        location = player['location']
        playerStats = [player['assists'], player['boost'], player['cartouches'], player['demos'], player['goals'],
                       location['x'], location['y'], location['z'], location['pitch'], location['roll'], location['yaw'],
                       player['saves'], player['score'], player['shots'], player['speed'], player['touches']]
        teamPlayerStats.append(playerStats)
    return score, torch.FloatTensor(teamPlayerStats)

class History:
    def __init__(self, times, ball, scores, team1, team2):
        self.times = times
        self.ball = ball
        self.scores = scores
        self.team1 = team1
        self.team2 = team2

class Match:
    def __init__(self, winner, history):
        self.winner = winner
        self.history = history

matches = []

for i in range(1, 39):
    f = open(f"..\\rocket-league-scribe\\recordings\\{i}.json", "r")
    doc = json.load(f)
    winner = torch.FloatTensor(doc['winner'])
    snapshots = doc['snapshots']
    times = []
    balls = []
    scores = []
    team1s = []
    team2s = []
    for snapshot in snapshots:
        time = torch.FloatTensor([snapshot['timeLeft'], snapshot['timePassed']])
        ball = snapshot['ball']
        ballStats = torch.FloatTensor([ball['speed'], ball['location']['x'], ball['location']['y'], ball['location']['z']])
        team1 = snapshot['team1']
        score1, team1Stats = getTeamStats(team1)
        team2 = snapshot['team2']
        score2, team2Stats = getTeamStats(team2)
        times.append(time)
        balls.append(ballStats)
        scores.append(torch.FloatTensor([score1, score2]))
        team1s.append(team1Stats)
        team2s.append(team2Stats)
    history = History(times, balls, scores, team1s, team2s)
    match = Match(winner, history)
    matches.append(match)

class PlayerRNN(nn.Module):
    def __init__(self, player_size, player_hidden_size):
        super(PlayerRNN, self).__init__()
        self.player_hidden_size = player_hidden_size
        self.num_layers = 1

        self.linear = nn.Linear(player_size, player_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(player_size, player_hidden_size)

    def forward(self, player, hidden):
        linear = self.relu(self.linear(player)).unsqueeze(0)
        return self.gru(linear, hidden)

    def init_hidden(self):
        return torch.zeros(self.num_layers, 3, self.player_hidden_size)

class TeamRNN(nn.Module):
    def __init__(self, player_size, player_hidden_size):
        super(TeamRNN, self).__init__()
        self.playerNet = PlayerRNN(player_size, player_hidden_size)
        self.conv = nn.Conv1d(player_size, player_size, 3, padding = 0)

    def forward(self, players, hiddens):
        playerEmbeds, hiddensNext = self.playerNet(players, hiddens) # uses batch dimension of 3 (# of players on a team)
        return self.conv(playerEmbeds.permute(0, 2, 1)).view(-1), hiddensNext # Now we treat the player_size dimension as the layers dimension (no batches)

    def init_hidden(self):
        return self.playerNet.init_hidden()

class GameNet(nn.Module):
    def __init__(self, player_size, player_hidden_size):
        super(GameNet, self).__init__()
        self.teamNet = TeamRNN(player_size, player_hidden_size)
        # self.playerNet = PlayerRNN(player_size, player_hidden_size)
        self.conv = nn.Conv1d(player_size + 1, player_size + 1, 2, padding = 0) # add one for the score number
        self.linear = nn.Linear(player_size + 1 + 4 + 2, 2) # 1 for score, 4 for ball, 2 for times
        self.softmax = nn.Softmax()

    def forward(self, team1, team2, hiddens1, hiddens2, scores, ball, times):
        teamEmbed1, hiddens1Next = self.teamNet(team1, hiddens1)
        team1WithScores = torch.cat((teamEmbed1, scores[0].unsqueeze(0)))
        team1WithScores.unsqueeze_(0)
        teamEmbed2, hiddens2Next = self.teamNet(team2, hiddens2)
        team2WithScores = torch.cat((teamEmbed2, scores[1].unsqueeze(0)))
        team2WithScores.unsqueeze_(0)
        combined = torch.cat((team1WithScores, team2WithScores))
        bothTeams = self.conv(combined.permute(1, 0)).view(-1)
        game = torch.cat((bothTeams, ball, times)) # cat with ball and times
        return self.softmax(self.linear(game)), hiddens1Next, hiddens2Next # Convolve the team embeds over the team dimension with player_size channels (no batches)

    def init_hidden(self):
        return self.teamNet.init_hidden()

""" hyperparameters """
player_hidden_size = 16

network = GameNet(matches[0].history.team1[0].shape[1], player_hidden_size)

testMatch = matches[0]
prediction, hiddens1, hiddens2 = network(testMatch.history.team1[0], testMatch.history.team2[0], network.init_hidden(), network.init_hidden(), testMatch.history.scores[0], testMatch.history.ball[0], testMatch.history.times[0])
print(prediction)

num_params = sum([np.prod(p.size()) for p in network.parameters()])
print(f"# Parameters: {num_params}")

# class RoLLMaPDataset(Dataset):
#     def __init__(self, data):
#         super().__init__()
#         self.data = []
#         for d in data:
#             self.data.append(d)

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)