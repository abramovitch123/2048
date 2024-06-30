class ScoreManager:
    def __init__(self):
        self.score = 0
        self.best = 0

    def check_highscore(self):
        if self.score >= self.best:
            self.best = self.score

    def reset_score(self):
        self.score = 0
