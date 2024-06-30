import os
import pygame
import sys
from constants import SCREEN_COLOR
from game import Game

WIDTH, HEIGHT = 900, 900

def center_window(width, height):
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{(pygame.display.Info().current_w - width) // 2},{(pygame.display.Info().current_h - height) // 2}"

def main():
    pygame.init()
    pygame.display.set_caption('2048')

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    center_window(WIDTH, HEIGHT)
    screen.fill(SCREEN_COLOR)

    game = Game(screen, WIDTH, HEIGHT)
    game.show_initial_menu()

    clock = pygame.time.Clock()
    fps = 60

    try:
        while True:
            screen.fill(SCREEN_COLOR)

            if game.show_menu:
                game.gui.show_start()
                game.gui.show_initial_menu()
            else:
                game.gui.show_start()
                game.draw_board(game.tiles_dqn)
                game.gui.show_controls()
                game.gui.menu.show()
                game.gui.update_scores(game.score_manager[game.current_mode].score, game.score_manager[game.current_mode].best)
                game.gui.show_menu_button()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.save_best_scores()
                    pygame.quit()
                    sys.exit()
                if game.show_menu and event.type == pygame.MOUSEBUTTONDOWN:
                    action = game.gui.initial_menu_listener(event)
                    if action == 'dqn':
                        game.start_dqn_mode()
                    elif action == 'dataset':
                        game.start_dataset_mode()
                    elif action == 'best_dqn':
                        game.start_best_dqn_mode()
                    elif action == 'best_dataset':
                        game.start_best_dataset_mode()
                elif not game.show_menu:
                    if event.type == pygame.KEYDOWN:
                        game.handle_key_event(event)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        action = game.gui.action_listener(event)
                        game.handle_mouse_event(action)

            if not game.show_menu and game.playing:
                game.update_game_state()

            game.update_display()
            clock.tick(fps)
    except KeyboardInterrupt:
        game.save_best_scores()
        pygame.quit()
        sys.exit()

    print(f"Best DQN Score: {game.best_score['best_dqn']}")
    print(f"Best Dataset Score: {game.best_score['best_dataset']}")

if __name__ == "__main__":
    main()
