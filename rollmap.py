import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

def getTeamStats(team):
    score = team['score']
    teamPlayerStats = []
    for player in team['players']:
        location = player['location']
        playerStats = [player['assists'], player['boost'], player['cartouches'], player['demos'], player['goals'],
                       location['x'], location['y'], location['z'], location['pitch'], location['roll'], location['yaw'],
                       player['saves'], player['score'], player['shots'], player['speed'], player['touches']]
        teamPlayerStats.append(playerStats)
    return score, teamPlayerStats

class History:
    def __init__(self, times, ball, scores, team1, team2):
        self.times = times
        self.ball = ball
        self.scores = scores
        self.team1 = team1
        self.team2 = team2

    def mirrorTeams(self):
        times = self.times # reference
        ball = torch.clone(self.ball) # deep copy
        ball[:, [1, 2]] *= -1
        scores = self.scores[:, [1, 0]] # view
        team1 = torch.clone(self.team2)
        team1[:, :, [5, 6]] *= -1
        team1[:, :, 10] += 1
        condition = team1[:, :, 10] > 1
        team1[:, :, 10] -= 2 * condition
        team2 = torch.clone(self.team1)
        team2[:, :, [5, 6]] *= -1
        team2[:, :, 10] += 1
        condition = team2[:, :, 10] > 1
        team2[:, :, 10] -= 2 * condition

        return History(times, ball, scores, team1, team2)
    
    def getPlayerPermute(self, num): # num in [0,5]
        if num == 0:
            return [0, 1, 2]
        elif num == 1:
            return [0, 2, 1]
        elif num == 2:
            return [1, 0, 2]
        elif num == 3:
            return [1, 2, 0]
        elif num == 4:
            return [2, 0, 1]
        else:
            return [2, 1, 0]
    
    def permutePlayers(self, num): # num in [0,35]
        team1Order = num % 6
        team2Order = num // 6
        return History(self.times, self.ball, self.scores, self.team1[:, self.getPlayerPermute(team1Order), :], self.team2[:, self.getPlayerPermute(team2Order), :])


class Match:
    def __init__(self, winner, history):
        self.winner = winner
        self.history = history
    
    def permute(self, num): # num in [0,71]
        if num == 0:
            return self
        playerOrdering = num % 36
        teamOrdering = num // 36
        winner = self.winner
        history = History(torch.clone(self.history.times), torch.clone(self.history.ball), torch.clone(self.history.scores), torch.clone(self.history.team1), torch.clone(self.history.team2))
        if teamOrdering == 1:
            winner = (winner + 1) % 2
            # mirror ball, scores, and player positions for each item in history
            history = history.mirrorTeams()
        history = history.permutePlayers(playerOrdering)
        return Match(winner, history)

matches = []

for i in range(1, 39):
    f = open(f"..\\rocket-league-scribe\\recordings\\{i}.json", "r")
    doc = json.load(f)
    winner = torch.IntTensor([doc['winner']])
    snapshots = doc['snapshots']
    times = []
    balls = []
    scores = []
    team1s = []
    team2s = []
    for snapshot in snapshots:
        time = [snapshot['timeLeft'], snapshot['timePassed']]
        ball = snapshot['ball']
        ballStats = [ball['speed'], ball['location']['x'], ball['location']['y'], ball['location']['z']]
        team1 = snapshot['team1']
        score1, team1Stats = getTeamStats(team1)
        team2 = snapshot['team2']
        score2, team2Stats = getTeamStats(team2)
        times.append(time)
        balls.append(ballStats)
        scores.append([score1, score2])
        team1s.append(team1Stats)
        team2s.append(team2Stats)
    history = History(torch.FloatTensor(times), torch.FloatTensor(balls), torch.FloatTensor(scores), torch.FloatTensor(team1s), torch.FloatTensor(team2s))
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
        self.conv = nn.Conv1d(player_hidden_size, player_hidden_size, 3, padding = 0)

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
        self.conv = nn.Conv1d(player_hidden_size + 1, player_hidden_size + 1, 2, padding = 0) # add one for the score number
        self.linear = nn.Linear(player_hidden_size + 1 + 4 + 2, 2) # 1 for score, 4 for ball, 2 for times
        self.softmax = nn.Softmax(dim=0)

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

class WinnerLoss(nn.Module):
    def __init__(self):
        super(WinnerLoss, self).__init__()
        self.lossModule = nn.CrossEntropyLoss()
        self.results = torch.FloatTensor([[1,0],[0,1]])

    def forward(self, prediction, winner):
        return self.lossModule(prediction, self.results[winner].view(-1))
    
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.lossModule = nn.CrossEntropyLoss()
        self.expected = torch.FloatTensor([0.5,0.5])

    def forward(self, prediction):
        return self.lossModule(prediction, self.expected)

class RoLLMaPDataset(Dataset):
    def __init__(self, matches):
        super().__init__()
        self.matches = []
        for m in matches:
            self.matches.append(m)

    def __getitem__(self, index):
        return self.matches[index % len(self.matches)].permute(index // len(self.matches))

    def __len__(self):
        return len(self.matches) * 72 # 6 ways you can order team 1 players * 6 ways you can order team 2 players * 2 ways you can order the teams
    


""" hyperparameters """
player_hidden_size = 200
learning_rate = 1e-5
symmetry_step_every_n_steps = 10
symmetry_train_steps = 10

network = GameNet(matches[0].history.team1[0].shape[1], player_hidden_size)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
dataset = RoLLMaPDataset(matches)
winnerLoss = WinnerLoss()
uncertaintyLoss = UncertaintyLoss()
crossEntropyLoss = nn.CrossEntropyLoss()

num_params = sum([np.prod(p.size()) for p in network.parameters()])
print(f"# Parameters: {num_params}")

print(f"# of items in dataset: {len(dataset)}")

def learn_symmetry(network, lossFn, optimizer):
    # lossFn.zero_grad()
    index = random.randint(0, len(dataset) // 72 - 1)
    match = dataset[index]
    permuteNum = random.randint(1, 71)
    flipPrediction = permuteNum // 36 == 1
    permutation = match.permute(permuteNum)
    hiddens1, hiddens2, hiddens3, hiddens4 = network.init_hidden(), network.init_hidden(), network.init_hidden(), network.init_hidden()
    timeSteps = len(match.history.times)
    # totalLoss = torch.tensor(0)
    for i in range(timeSteps):
        lossFn.zero_grad()
        prediction1, hiddens1, hiddens2 = network(match.history.team1[i], match.history.team2[i], hiddens1, hiddens2, match.history.scores[i], match.history.ball[i], match.history.times[i])
        prediction2, hiddens3, hiddens4 = network(permutation.history.team1[i], permutation.history.team2[i], hiddens3, hiddens4, permutation.history.scores[i], permutation.history.ball[i], permutation.history.times[i])
        if flipPrediction:
            prediction2 = prediction2[[1, 0]]
        # totalLoss += lossFn(prediction1, prediction2)
        loss = lossFn(prediction1, prediction2)
        if (i + 1) % symmetry_step_every_n_steps == 0 or i == timeSteps - 1:
            loss.backward()
            hiddens1.detach_()
            hiddens2.detach_()
            hiddens3.detach_()
            hiddens4.detach_()
            optimizer.step()
    # totalLoss.backward()
    # optimizer.step()

while True:
    for _ in range(symmetry_train_steps):
        learn_symmetry(network, crossEntropyLoss, optimizer)
    # do normal training here
    print("normal train")

# testMatch = matches[0]
# prediction, hiddens1, hiddens2 = network(testMatch.history.team1[0], testMatch.history.team2[0], network.init_hidden(), network.init_hidden(), testMatch.history.scores[0], testMatch.history.ball[0], testMatch.history.times[0])
# print(prediction)

# i = 0
# while True:
#     timeLeft = testMatch.history.times[150][0]
#     timePassed = testMatch.history.times[150][1]
#     loss = uncertaintyLoss(prediction)*timeLeft + winnerLoss(prediction, testMatch.winner)*timePassed
#     print(loss)
#     if loss < 0.1:
#         break
#     loss.backward()
#     optimizer.step()

#     optimizer.zero_grad()
#     i = (i + 1) % len(matches)
#     testMatch = matches[i]
#     prediction, hiddens1, hiddens2 = network(testMatch.history.team1[150], testMatch.history.team2[150], network.init_hidden(), network.init_hidden(), testMatch.history.scores[150], testMatch.history.ball[150], testMatch.history.times[150])
#     print(prediction)