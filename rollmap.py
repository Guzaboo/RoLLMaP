import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

device = 'cuda'

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
            match = self.mirror()
            winner = match.winner
            history = match.history
        history = history.permutePlayers(playerOrdering)
        return Match(winner, history)
    
    def mirror(self):
        # mirror ball, scores, and player positions for each item in history
        return Match((self.winner + 1) % 2, self.history.mirrorTeams())

matches = []

for i in range(1, 60):
    f = open(f"..\\rocket-league-scribe\\recordings\\{i}.json", "r")
    doc = json.load(f)
    winner = torch.tensor([doc['winner']], dtype=torch.int32, device=device)
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
    history = History(torch.tensor(times, dtype=torch.float32, device=device),
                      torch.tensor(balls, dtype=torch.float32, device=device),
                      torch.tensor(scores, dtype=torch.float32, device=device),
                      torch.tensor(team1s, dtype=torch.float32, device=device),
                      torch.tensor(team2s, dtype=torch.float32, device=device))
    match = Match(winner, history)
    matches.append(match)

class PlayerRNN(nn.Module):
    def __init__(self, player_size, player_hidden_size):
        super(PlayerRNN, self).__init__()
        self.player_hidden_size = player_hidden_size
        self.num_layers = 1

        self.linear = nn.Linear(player_size, player_size, device=device)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(player_size, player_hidden_size, device=device)

    def forward(self, player, hidden):
        linear = self.relu(self.linear(player)).unsqueeze(0)
        return self.gru(linear, hidden)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.player_hidden_size, device=device)

class TeamNet(nn.Module):
    def __init__(self, player_size, player_hidden_size):
        super(TeamNet, self).__init__()
        self.playerNet = PlayerRNN(player_size, player_hidden_size)
        self.linear = nn.Linear(player_hidden_size * 3, player_hidden_size, device=device)
        self.relu = nn.ReLU()

    def forward(self, players, hiddens):
        playerEmbeds = []
        nextHiddens = []
        for player, hidden in zip(players.permute(1, 0, 2), hiddens.permute(2, 0, 1, 3)):
            playerEmbed, nextHidden = self.playerNet(player, hidden)
            playerEmbeds.append(playerEmbed.squeeze(0))
            nextHiddens.append(nextHidden)

        allPlayers = []
        allPlayers.append(torch.cat((playerEmbeds[0], playerEmbeds[1], playerEmbeds[2]), 1))
        allPlayers.append(torch.cat((playerEmbeds[0], playerEmbeds[2], playerEmbeds[1]), 1))
        allPlayers.append(torch.cat((playerEmbeds[1], playerEmbeds[0], playerEmbeds[2]), 1))
        allPlayers.append(torch.cat((playerEmbeds[1], playerEmbeds[2], playerEmbeds[0]), 1))
        allPlayers.append(torch.cat((playerEmbeds[2], playerEmbeds[0], playerEmbeds[1]), 1))
        allPlayers.append(torch.cat((playerEmbeds[2], playerEmbeds[1], playerEmbeds[0]), 1))

        teamEmbed = self.relu(torch.mean(torch.stack([self.linear(concatenated) for concatenated in allPlayers]), 0))

        return teamEmbed, torch.stack(nextHiddens).permute(1, 2, 0, 3)

    def init_hidden(self, batch_size):
        return torch.stack([self.playerNet.init_hidden(batch_size) for _ in range(3)]).permute(1, 2, 0, 3)

class GameNet(nn.Module):
    def __init__(self, player_size, player_hidden_size):
        super(GameNet, self).__init__()
        team_embed_size = player_hidden_size
        self.teamNet = TeamNet(player_size, team_embed_size)
        self.linear = nn.Linear(2*(team_embed_size + 1) + 4 + 2, 2, device=device) # 1 for score, 4 for ball, 2 for times
        self.softmax = nn.Softmax(dim=1)

    def forward(self, team1, team2, hiddens1, hiddens2, scores, ball, times):
        teamEmbed1, hiddens1Next = self.teamNet(team1, hiddens1)
        team1WithScores = torch.cat((teamEmbed1, scores[:,0].unsqueeze(1)), 1)
        teamEmbed2, hiddens2Next = self.teamNet(team2, hiddens2)
        team2WithScores = torch.cat((teamEmbed2, scores[:,1].unsqueeze(1)), 1)
        combined1 = torch.cat((team1WithScores, team2WithScores), 1)
        combined2 = torch.cat((team2WithScores, team1WithScores), 1)
        rotatedBall = torch.clone(ball) # TODO - rotate ball
        rotatedBall[:, [1, 2]] *= -1
        games = []
        games.append(torch.cat((combined1, ball, times), 1)) # cat with ball and times
        games.append(torch.cat((combined2, rotatedBall, times), 1))
        processedGames = torch.stack([self.linear(game) for game in games])
        processedGames[1,:,[0,1]] = processedGames[1,:,[1,0]]
        return self.softmax(torch.mean(processedGames, 0)), hiddens1Next, hiddens2Next # Convolve the team embeds over the team dimension with player_size channels (no batches)

    def init_hidden(self, batch_size):
        return self.teamNet.init_hidden(batch_size)

class WinnerLoss(nn.Module):
    def __init__(self):
        super(WinnerLoss, self).__init__()
        self.lossModule = nn.CrossEntropyLoss()
        self.results = torch.tensor([[1,0],[0,1]], dtype=torch.float32, device=device)

    def forward(self, prediction, winner):
        return self.lossModule(prediction, self.results[winner])
        # return torch.mean(self.results[(winner + 1) % 2] * prediction)

class RoLLMaPDataset(Dataset):
    def __init__(self, matches):
        super().__init__()
        self.matches = []
        for m in matches:
            self.matches.append(m)

    def __getitem__(self, index):
        return self.matches[index % len(self.matches)].permute(index // len(self.matches))

    def __len__(self):
        return len(self.matches)
    


""" hyperparameters """
player_hidden_size = 20
learning_rate = 1e-4
batch_size = 7
test_batch_size = 3
percent_training_data = .90
num_batches = 50

network = GameNet(matches[0].history.team1[0].shape[1], player_hidden_size)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
dataset = RoLLMaPDataset(matches)
winnerLoss = WinnerLoss()

num_params = sum([np.prod(p.size()) for p in network.parameters()])
print(f"# Parameters: {num_params}")

print(f"# of items in dataset: {len(dataset)}")

def train_model(network, lossFn, optimizer):
    matches = [dataset[i] for i in random.sample(range(0, math.floor(len(dataset) * percent_training_data)), batch_size)]
    num_forced_winner1 = batch_size // 3
    num_forced_winner2 = batch_size // 3
    for i in range(num_forced_winner1):
        if matches[i].winner != 0:
            matches[i] = matches[i].mirror()
    for i in range(num_forced_winner1, num_forced_winner1 + num_forced_winner2):
        if matches[i].winner != 1:
            matches[i] = matches[i].mirror()

    winners = torch.stack([match.winner for match in matches]).squeeze(1)
    hiddens1 = network.init_hidden(batch_size)
    hiddens2 = network.init_hidden(batch_size)
    maxTimesteps = max([len(match.history.times) for match in matches])
    totalLoss = torch.tensor(0, dtype=torch.float32, device=device)
    numLossCalculations = 0
    for i in range(maxTimesteps):
        team1s = torch.stack([match.history.team1[i if i < len(match.history.times) else -1] for match in matches], dim=0)
        team2s = torch.stack([match.history.team2[i if i < len(match.history.times) else -1] for match in matches], dim=0)
        scores = torch.stack([match.history.scores[i if i < len(match.history.times) else -1] for match in matches], dim=0)
        balls = torch.stack([match.history.ball[i if i < len(match.history.times) else -1] for match in matches], dim=0)
        times = torch.stack([match.history.times[i if i < len(match.history.times) else -1] for match in matches], dim=0)
        predictions, hiddens1, hiddens2 = network(team1s, team2s, hiddens1, hiddens2, scores, balls, times)
        if i % 10 == 9 or i == maxTimesteps - 1:
            loss = lossFn(predictions, winners)
            totalLoss += loss
            numLossCalculations += 1
            loss.backward()
            hiddens1 = hiddens1.detach()
            hiddens2 = hiddens2.detach()
            optimizer.step()
    return totalLoss.item() / numLossCalculations

def test_model(network, lossFn, drawGraph):
    with torch.no_grad():
        indices = random.sample(range(math.floor(len(dataset) * percent_training_data), len(dataset)), test_batch_size)
        matches = [dataset[i] for i in indices]

        winners = torch.stack([match.winner for match in matches]).squeeze(1)
        hiddens1 = network.init_hidden(test_batch_size)
        hiddens2 = network.init_hidden(test_batch_size)
        maxTimesteps = max([len(match.history.times) for match in matches])
        totalLoss = torch.tensor(0, dtype=torch.float32, device=device)
        predictionsHistory = [[] for _ in range(test_batch_size)]
        prevScores = [[0,0] for _ in range(test_batch_size)]
        scoreUpdates = [[] for _ in range(test_batch_size)]
        for i in range(maxTimesteps):
            team1s = torch.stack([match.history.team1[i if i < len(match.history.times) else -1] for match in matches], dim=0)
            team2s = torch.stack([match.history.team2[i if i < len(match.history.times) else -1] for match in matches], dim=0)
            scores = torch.stack([match.history.scores[i if i < len(match.history.times) else -1] for match in matches], dim=0)
            balls = torch.stack([match.history.ball[i if i < len(match.history.times) else -1] for match in matches], dim=0)
            times = torch.stack([match.history.times[i if i < len(match.history.times) else -1] for match in matches], dim=0)

            for j in range(test_batch_size):
                if scores[j][0] != prevScores[j][0] or scores[j][1] != prevScores[j][1]:
                    prevScores[j][0] = scores[j][0].item()
                    prevScores[j][1] = scores[j][1].item()
                    scoreUpdates[j].append((i, prevScores[j][0], prevScores[j][1]))

            predictions, hiddens1, hiddens2 = network(team1s, team2s, hiddens1, hiddens2, scores, balls, times)
            for i, prediction in enumerate(predictions):
                predictionsHistory[i].append(prediction[0].item())
            loss = lossFn(predictions, winners)
            totalLoss += loss
        if drawGraph:
            for i, predictionHistory in enumerate(predictionsHistory):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for scoreUpdate in scoreUpdates[i]:
                    plt.axvline(x=scoreUpdate[0], color='lightgray')
                    plt.text(scoreUpdate[0], predictionHistory[scoreUpdate[0]] + 0.05 if predictionHistory[scoreUpdate[0]] + 0.05 <= 0.95 else predictionHistory[scoreUpdate[0]] - 0.05, f'{int(scoreUpdate[1])}-{int(scoreUpdate[2])}', rotation=0, va='bottom', ha='center')
                ax.plot(range(len(matches[i].history.times)), predictionHistory[0:len(matches[i].history.times)], label='% chance of team 1 winning')
                plt.title(f'{indices[i] + 1}.json')
                plt.xlabel('Timestep')
                plt.ylabel('% chance of team 1 winning')
                plt.ylim(0, 1)
                plt.xlim(0, len(matches[i].history.times))
                plt.axhline(y=0.5, color='gray', linestyle='--')
                plt.show()
        return totalLoss.item() / maxTimesteps

train_losses = []
test_losses = []

batches_trained = 0
while batches_trained < num_batches:
    train_losses.append(train_model(network, winnerLoss, optimizer))
    batches_trained += 1
    print(f"Trained on {batches_trained} batches")
    test_losses.append(test_model(network, winnerLoss, batches_trained % 10 == 0))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(num_batches), train_losses, label='Training Loss')
ax.plot(range(num_batches), test_losses, label='Testing Loss')
plt.title(f'Loss')
plt.xlabel('# batches used for training')
plt.ylabel('Average loss')
plt.legend()
plt.show()