import json

f = open("..\\rocket-league-scribe\\recordings\\1.json", "r")
# print(f.read())

doc = json.load(f)
winner = doc['winner']
history = doc['snapshots']
for snapshot in history:
    time = [snapshot['timeLeft'], snapshot['timePassed']]
    ball = snapshot['ball']
    team1 = snapshot['team1']
    team2 = snapshot['team2']
    if snapshot == history[150]:
        print(time)
        print(team1)
        print(team2)
        print(ball)
# print(history[0])