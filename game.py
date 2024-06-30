import os
import time
import pygame
import random
import numpy as np
from constants import *
from dqn_agent import DQNAgent
from dataset import DatasetAgent
from score_manager import ScoreManager
from ui import GUI

class Game:
    def __init__(self, screen, WIDTH, HEIGHT):
        self.screen = screen
        self.tiles_dqn = np.zeros((ROWS, COLS))
        self.tiles_dataset = np.zeros((ROWS, COLS))
        self.gui = GUI(screen, WIDTH, HEIGHT)
        self.score_manager = {
            'dqn': ScoreManager(),
            'dataset': ScoreManager(),
            'best_dqn': ScoreManager(),
            'best_dataset': ScoreManager()
        }
        self.current_mode = 'dqn'
        self.lbl_font = pygame.font.SysFont('verdana', 40, bold=True)
        self.generate = False
        self.playing = True
        self.ai_active = False
        self.show_menu = True
        self.state_size = ROWS * COLS
        self.action_size = 4
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.dataset_agent = DatasetAgent(self.state_size, self.action_size)
        self.best_score = {'dqn': 0, 'dataset': 0, 'best_dqn': 0, 'best_dataset': 0}
        self.load_best_scores()
        self.track_scores()
        self.reset_game()

    def load_best_scores(self):
        if os.path.exists('best_dqn_score.txt'):
            with open('best_dqn_score.txt', 'r') as f:
                self.best_score['best_dqn'] = float(f.read())
        if os.path.exists('best_dataset_score.txt'):
            with open('best_dataset_score.txt', 'r') as f:
                self.best_score['best_dataset'] = float(f.read())
        print(f"Loaded best scores: DQN: {self.best_score['best_dqn']}, Dataset: {self.best_score['best_dataset']}")

    def save_best_scores(self):
        with open('best_dqn_score.txt', 'w') as f:
            f.write(str(self.best_score['best_dqn']))
        with open('best_dataset_score.txt', 'w') as f:
            f.write(str(self.best_score['best_dataset']))
        print(f"Saved best scores: DQN: {self.best_score['best_dqn']}, Dataset: {self.best_score['best_dataset']}")

    def get_best_scores(self):
        best_scores = {}
        if os.path.exists('best_dqn_score.txt'):
            with open('best_dqn_score.txt', 'r') as f:
                best_scores['best_dqn'] = float(f.read())
        else:
            best_scores['best_dqn'] = 0

        if os.path.exists('best_dataset_score.txt'):
            with open('best_dataset_score.txt', 'r') as f:
                best_scores['best_dataset'] = float(f.read())
        else:
            best_scores['best_dataset'] = 0

        return best_scores

    def track_scores(self):
        self.scores = {'dqn': [], 'dataset': [], 'best_dqn': [], 'best_dataset': []}

    def update_scores(self):
        if self.current_mode in self.scores:
            self.scores[self.current_mode].append(self.score_manager[self.current_mode].score)

    def reset_game(self):
        self.tiles_dqn = np.zeros((ROWS, COLS))
        self.tiles_dataset = np.zeros((ROWS, COLS))
        self.score_manager[self.current_mode].reset_score()
        self.generate_starting_tiles()
        self.playing = True
        self.generate = False

    def generate_starting_tiles(self):
        positions = random.sample([(row, col) for row in range(ROWS) for col in range(COLS)], 2)
        values = [2, random.choice([2, 4])]
        for pos, value in zip(positions, values):
            self.tiles_dqn[pos[0], pos[1]] = value
            self.tiles_dataset[pos[0], pos[1]] = value

    def new(self, auto_start_ai=False, load_models=False):
        self.reset_game()
        self.gui.show_start()
        self.ai_active = auto_start_ai
        if load_models:
            if self.current_mode == 'best_dqn':
                self.agent.load_best_model(file_path='best_dqn_model.pth')
            elif self.current_mode == 'best_dataset':
                self.dataset_agent.load_best_model(file_path='best_dataset_model.pth')
            print("Loaded best models for AI.")

    def get_state(self, tiles):
        return tiles.flatten()

    def choose_action(self, state, mode):
        if mode == 'dataset' or mode == 'best_dataset':
            return self.dataset_agent.act(state)
        return self.agent.act(state)

    def remember(self, state, action, reward, next_state, done, mode):
        reward = reward if reward is not None else 0
        if mode == 'dataset' or mode == 'best_dataset':
            self.dataset_agent.remember(state, action)
        else:
            self.agent.remember(state, action, reward, next_state, done)

    def replay(self, mode):
        if mode == 'dataset' or mode == 'best_dataset':
            self.dataset_agent.train()
        else:
            self.agent.replay()

    def is_models_trained(self):
        return os.path.exists('best_dqn_model.pth') and os.path.exists('best_dataset_model.pth')

    def show_training_required_message(self):
        print("You need to train both DQN and Dataset models before playing in BEST mode.")

    def start_dqn_mode(self):
        self.current_mode = 'dqn'
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.agent.epsilon = 1.0  # Ensure DQN starts with high exploration
        print("Starting DQN mode.")
        self.new(auto_start_ai=True, load_models=False)
        self.show_menu = False

    def start_dataset_mode(self):
        self.current_mode = 'dataset'
        self.dataset_agent = DatasetAgent(self.state_size, self.action_size)
        print("Starting Dataset mode.")
        self.new(auto_start_ai=True, load_models=False)
        self.ai_active = True
        self.show_menu = False

    def start_best_dqn_mode(self):
        self.current_mode = 'best_dqn'
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.agent.load_best_model(file_path='best_dqn_model.pth')
        print(f"Starting Best DQN mode with best score: {self.agent.best_score}")
        self.new(auto_start_ai=True, load_models=True)
        self.show_menu = False

    def start_best_dataset_mode(self):
        self.current_mode = 'best_dataset'
        self.dataset_agent = DatasetAgent(self.state_size, self.action_size)
        self.dataset_agent.load_best_model(file_path='best_dataset_model.pth')
        print(f"Starting Best Dataset mode with best score: {self.dataset_agent.best_score}")
        self.new(auto_start_ai=True, load_models=True)
        self.ai_active = True
        self.show_menu = False

    def show_initial_menu(self):
        self.show_menu = True

    def handle_key_event(self, event):
        if event.key == pygame.K_UP:
            self.slide_tiles('UP', self.tiles_dqn, self.current_mode)
        elif event.key == pygame.K_DOWN:
            self.slide_tiles('DOWN', self.tiles_dqn, self.current_mode)
        elif event.key == pygame.K_RIGHT:
            self.slide_tiles('RIGHT', self.tiles_dqn, self.current_mode)
        elif event.key == pygame.K_LEFT:
            self.slide_tiles('LEFT', self.tiles_dqn, self.current_mode)
        if self.generate:
            self.generate_tiles()
            self.generate = False

    def handle_mouse_event(self, action):
        if action == 'ai_toggle':
            self.ai_active = not self.ai_active
        elif action == 'try_again':
            self.new(auto_start_ai=True)
            self.gui.show_start()
        elif action == 'new_game':
            self.new(auto_start_ai=False)
            self.gui.show_start()
        elif action == 'menu':
            self.show_initial_menu()

    def update_game_state(self):
        if self.playing:
            if self.ai_active:
                self.update_ai(self.tiles_dqn, self.current_mode)

        if not self.show_menu and self.is_game_over(self.tiles_dqn) and self.playing:
            self.playing = False
            self.gui.menu.active = True

    def update_display(self):
        self.score_manager[self.current_mode].check_highscore()
        pygame.display.update()

    def slide_tiles(self, direction, tiles, mode):
        moved = False
        if direction == 'UP':
            for row in range(1, ROWS):
                for col in range(COLS):
                    if tiles[row][col] != 0:
                        moved = self.__move_and_merge(direction, row, col, tiles, mode) or moved
        if direction == 'DOWN':
            for row in range(ROWS - 2, -1, -1):
                for col in range(COLS):
                    if tiles[row][col] != 0:
                        moved = self.__move_and_merge(direction, row, col, tiles, mode) or moved
        if direction == 'RIGHT':
            for row in range(ROWS):
                for col in range(COLS - 2, -1, -1):
                    if tiles[row][col] != 0:
                        moved = self.__move_and_merge(direction, row, col, tiles, mode) or moved
        if direction == 'LEFT':
            for row in range(ROWS):
                for col in range(1, COLS):
                    if tiles[row][col] != 0:
                        moved = self.__move_and_merge(direction, row, col, tiles, mode) or moved
        return moved

    def __move_and_merge(self, direction, row, col, tiles, mode):
        dx, dy = 0, 0
        if direction == 'UP':
            dy = -1
        elif direction == 'DOWN':
            dy = 1
        elif direction == 'RIGHT':
            dx = 1
        elif direction == 'LEFT':
            dx = -1
        moved = False
        try:
            if tiles[row + dy][col + dx] == 0:
                tiles[row + dy][col + dx] = tiles[row][col]
                tiles[row][col] = 0
                self.generate = True
                moved = True
                self.__move_and_merge(direction, row + dy, col + dx, tiles, mode)
            elif tiles[row][col] == tiles[row + dy][col + dx]:
                tiles[row + dy][col + dx] *= 2
                tiles[row][col] = 0
                self.score_manager[mode].score += tiles[row + dy][col + dx]
                self.generate = True
                moved = True
        except IndexError:
            pass
        return moved

    def draw_board(self, tiles, x_offset=0):
        rShift, cShift = GAP, GAP
        pygame.draw.rect(self.screen, BOARD_COLOR, (
            x_offset + self.gui.board_rect[0] - GAP // 2,
            self.gui.board_rect[1] - GAP // 2,
            BOARD_WIDTH + GAP,
            BOARD_HEIGHT + GAP
        ))
        for row in range(ROWS):
            for col in range(COLS):
                tile_num = int(tiles[row][col])
                tile_color = TILES_COLORS[tile_num]
                rect = pygame.Rect(
                    x_offset + self.gui.board_rect[0] + cShift + col * TILE_SIZE,
                    self.gui.board_rect[1] + rShift + row * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(self.screen, tile_color, rect)
                tile_lbl_color = LBLS_COLORS[tile_num]
                if tile_num > 0:
                    lbl = self.lbl_font.render(str(tile_num), 0, tile_lbl_color)
                    lbl_rect = lbl.get_rect(center=rect.center)
                    self.screen.blit(lbl, lbl_rect)
                cShift += GAP
            rShift += GAP
            cShift = GAP

    def generate_tiles(self, first=False):
        empty_tiles = [(row, col) for row in range(ROWS) for col in range(COLS) if self.tiles_dqn[row][col] == 0]
        if empty_tiles:
            row, col = random.choice(empty_tiles)
            self.tiles_dqn[row][col] = 2 if first or random.randint(1, 10) <= 7 else 4
            self.tiles_dataset[row][col] = self.tiles_dqn[row][col]

    def is_game_over(self, tiles):
        return self.__full_board(tiles) and self.__no_more_moves(tiles)

    def __full_board(self, tiles):
        return all(tiles[row][col] != 0 for row in range(ROWS) for col in range(COLS))

    def __no_more_moves(self, tiles):
        for row in range(1, ROWS):
            for col in range(COLS):
                if tiles[row][col] == tiles[row - 1][col]:
                    return False
        for row in range(ROWS - 2, -1, -1):
            for col in range(COLS):
                if tiles[row][col] == tiles[row + 1][col]:
                    return False
        for row in range(ROWS):
            for col in range(COLS - 2, -1, -1):
                if tiles[row][col] == tiles[row][col + 1]:
                    return False
        for row in range(ROWS):
            for col in range(1, COLS):
                if tiles[row][col] == tiles[row][col - 1]:
                    return False
        return True

    def is_highest_tile_in_corner(self):
        corners = [self.tiles_dqn[0, 0], self.tiles_dqn[0, COLS - 1], self.tiles_dqn[ROWS - 1, 0], self.tiles_dqn[ROWS - 1, COLS - 1]]
        max_tile = np.max(self.tiles_dqn)
        return max_tile in corners

    def is_keeping_high_tiles_together(self, tiles):
        max_tile = np.max(tiles)
        positions = np.argwhere(tiles == max_tile)
        for pos in positions:
            row, col = pos
            if any(tiles[row + dr, col + dc] == max_tile for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= row + dr < ROWS and 0 <= col + dc < COLS):
                return True
        return False

    def calculate_reward(self, previous_tiles):
        reward = np.sum(self.tiles_dqn) - np.sum(previous_tiles)
        reward += np.max(self.tiles_dqn) * 2  # Encourage higher tiles

        if self.is_highest_tile_in_corner():
            reward += np.max(self.tiles_dqn) * 2

        filled_rows = sum(all(self.tiles_dqn[row][col] != 0 for col in range(COLS)) for row in range(ROWS))
        filled_cols = sum(all(self.tiles_dqn[row][col] != 0 for row in range(ROWS)) for col in range(COLS))
        reward += (filled_rows + filled_cols) * 2

        if self.is_highest_tile_in_corner():
            reward += 5

        if self.is_keeping_high_tiles_together(self.tiles_dqn):
            reward += 5

        if not self.is_highest_tile_in_corner():
            reward -= 5

        return reward

    def update_ai(self, tiles, mode):
        if self.playing:
            try:
                state = self.get_state(tiles)
                action = self.choose_action(state, mode)
                directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                direction = directions[action]

                previous_tiles = tiles.copy()
                moved = self.slide_tiles(direction, tiles, mode)
                if not moved:
                    alternative_directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
                    for alt_direction in alternative_directions:
                        if alt_direction != direction:
                            moved = self.slide_tiles(alt_direction, tiles, mode)
                            if moved:
                                break

                if moved and self.generate:
                    self.generate_tiles()
                    self.generate = False

                next_state = self.get_state(tiles)
                done = self.is_game_over(tiles)
                reward = self.calculate_reward(previous_tiles) if mode in ['dqn', 'best_dqn'] else None
                self.remember(state, action, reward, next_state, done, mode)
                self.replay(mode)

                if mode in ['dqn', 'best_dqn']:
                    print(f"Action: {direction}, Current Score: {self.score_manager[mode].score}, Reward: {reward}")
                else:
                    print(f"Action: {direction}, Current Score: {self.score_manager[mode].score}")

                if done:
                    print(f"Game Over! Final Score: {self.score_manager[mode].score}")
                    if self.score_manager[mode].score > self.best_score[mode]:
                        self.best_score[mode] = self.score_manager[mode].score
                        if mode in ['dataset', 'best_dataset']:
                            self.dataset_agent.save_best_model(file_path='best_dataset_model.pth')
                            if self.best_score[mode] > self.best_score['best_dataset']:
                                self.best_score['best_dataset'] = self.best_score[mode]
                                self.dataset_agent.save_best_model(file_path='best_dataset_model.pth')
                                print(f"New best dataset model saved with score: {self.best_score[mode]}")
                        elif mode in ['dqn', 'best_dqn']:
                            self.agent.save_best_model(file_path='best_dqn_model.pth')
                            if self.best_score[mode] > self.best_score['best_dqn']:
                                self.best_score['best_dqn'] = self.best_score[mode]
                                self.agent.save_best_model(file_path='best_dqn_model.pth')
                                print(f"New best DQN model saved with score: {self.best_score[mode]}")
                        self.save_best_scores()
                        print(f"New best scores saved with score: {self.best_score[mode]}")
                    self.playing = False
                    if self.ai_active:
                        self.new(auto_start_ai=True)
                        self.gui.show_start()
                    else:
                        time.sleep(0.1)
            except Exception as e:
                print(f"Error during AI update: {e}")

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('2048')

        game_count = 0
        running = True

        best_scores = self.get_best_scores()
        print(f"Best DQN Score: {best_scores['best_dqn']}")
        print(f"Best Dataset Score: {best_scores['best_dataset']}")

        while running:
            screen.fill(SCREEN_COLOR)
            if self.show_menu:
                self.gui.show_initial_menu()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_best_scores()
                        running = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if self.gui.dqn_btn_rect and self.gui.dqn_btn_rect.collidepoint(x, y):
                            self.start_dqn_mode()
                        elif self.gui.dataset_btn_rect and self.gui.dataset_btn_rect.collidepoint(x, y):
                            self.start_dataset_mode()
                        elif self.gui.best_dqn_btn_rect and self.gui.best_dqn_btn_rect.collidepoint(x, y):
                            self.start_best_dqn_mode()
                        elif self.gui.best_dataset_btn_rect and self.gui.best_dataset_btn_rect.collidepoint(x, y):
                            self.start_best_dataset_mode()
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_best_scores()
                        running = False
                    if event.type == pygame.KEYDOWN:
                        self.handle_key_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        action = self.gui.action_listener(event)
                        self.handle_mouse_event(action)

                if self.ai_active and self.playing:
                    self.update_ai(self.tiles_dqn, self.current_mode)

                if self.is_game_over(self.tiles_dqn) and self.playing:
                    self.playing = False
                    self.gui.menu.active = True

                self.draw_board(self.tiles_dqn)
                self.score_manager[self.current_mode].check_highscore()

                pygame.display.update()
                pygame.time.Clock().tick(60)

            if not self.playing:
                self.update_scores()
                game_count += 1

        pygame.quit()
        self.save_best_scores()

if __name__ == "__main__":
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('2048')
    game = Game(screen, WIDTH, HEIGHT)
    best_scores = game.get_best_scores()
    print(f"Best DQN Score: {best_scores['best_dqn']}")
    print(f"Best Dataset Score: {best_scores['best_dataset']}")
    game.run()
