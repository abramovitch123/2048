import pygame
from constants import *

class GUI:
    def __init__(self, screen, WIDTH, HEIGHT):
        self.screen = screen


        try:
            self.logo = pygame.image.load('images/logo.png')
            self.logo = pygame.transform.scale(self.logo, (self.logo.get_width() // 2, self.logo.get_height() // 2))
        except pygame.error as e:
            print(f"Error loading logo image: {e}")
            self.logo = None
        self.logo_pos = (WIDTH // 2 - (self.logo.get_width() // 2 if self.logo else 0), 20)

        try:
            self.message = pygame.image.load('images/message.png')
            self.message = pygame.transform.scale(self.message, (self.message.get_width() // 2, self.message.get_height() // 2))
        except pygame.error as e:
            print(f"Error loading message image: {e}")
            self.message = None
        self.message_pos = (WIDTH // 2 - (self.message.get_width() // 2 if self.message else 0), self.logo_pos[1] + (self.logo.get_height() if self.logo else 0) + 10)

        try:
            self.score_rect = pygame.image.load('images/score_rect.png')
            self.score_rect = pygame.transform.scale(self.score_rect, (90, 42))
        except pygame.error as e:
            print(f"Error loading score_rect image: {e}")
            self.score_rect = None
        self.score_rect_pos = (WIDTH // 2 - 50 - (self.score_rect.get_width() // 2 if self.score_rect else 0), self.message_pos[1] + (self.message.get_height() if self.message else 0) + 10)

        try:
            self.best_rect = pygame.image.load('images/best_rect.png')
            self.best_rect = pygame.transform.scale(self.best_rect, (90, 42))
        except pygame.error as e:
            print(f"Error loading best_rect image: {e}")
            self.best_rect = None
        self.best_rect_pos = (WIDTH // 2 + 50 - (self.best_rect.get_width() // 2 if self.best_rect else 0), self.message_pos[1] + (self.message.get_height() if self.message else 0) + 10)

        self.score_font = pygame.font.SysFont('verdana', 15, bold=True)

        try:
            self.newgame_btn = pygame.image.load('images/newgame_btn.png')
            self.newgame_btn = pygame.transform.scale(self.newgame_btn, (115, 40))
        except pygame.error as e:
            print(f"Error loading newgame_btn image: {e}")
            self.newgame_btn = None
        self.newgame_btn_pos = (WIDTH // 2 - (self.newgame_btn.get_width() // 2 if self.newgame_btn else 0), self.score_rect_pos[1] + 60)
        self.newgame_btn_rect = self.newgame_btn.get_rect(topleft=self.newgame_btn_pos) if self.newgame_btn else pygame.Rect(self.newgame_btn_pos, (115, 40))

        try:
            self.ai_btn = pygame.image.load('images/ai_btn.png')
            self.ai_btn = pygame.transform.scale(self.ai_btn, (115, 40))
        except pygame.error as e:
            print(f"Error loading ai_btn image: {e}")
            self.ai_btn = None
        self.ai_btn_pos = (WIDTH // 2 - (self.ai_btn.get_width() // 2 if self.ai_btn else 0), HEIGHT // 2 + BOARD_HEIGHT // 2 + 20)
        self.ai_btn_rect = self.ai_btn.get_rect(topleft=self.ai_btn_pos) if self.ai_btn else pygame.Rect(self.ai_btn_pos, (115, 40))

        self.board_rect = (WIDTH // 2 - BOARD_WIDTH // 2, HEIGHT // 2 - BOARD_HEIGHT // 2, BOARD_WIDTH, BOARD_HEIGHT)

        self.menu = Menu(screen)

        try:
            self.dqn_btn = pygame.image.load('images/dqn.png')
            self.dqn_btn = pygame.transform.scale(self.dqn_btn, (150, 60))
        except pygame.error as e:
            print(f"Error loading dqn_btn image: {e}")
            self.dqn_btn = None
        self.dqn_btn_pos = (WIDTH // 2 - (self.dqn_btn.get_width() // 2 if self.dqn_btn else 0), HEIGHT // 2 - 130)
        self.dqn_btn_rect = self.dqn_btn.get_rect(topleft=self.dqn_btn_pos) if self.dqn_btn else pygame.Rect(self.dqn_btn_pos, (150, 60))

        try:
            self.dataset_btn = pygame.image.load('images/dataset.png')
            self.dataset_btn = pygame.transform.scale(self.dataset_btn, (150, 60))
        except pygame.error as e:
            print(f"Error loading dataset_btn image: {e}")
            self.dataset_btn = None
        self.dataset_btn_pos = (WIDTH // 2 - (self.dataset_btn.get_width() // 2 if self.dataset_btn else 0), HEIGHT // 2 - 60)
        self.dataset_btn_rect = self.dataset_btn.get_rect(topleft=self.dataset_btn_pos) if self.dataset_btn else pygame.Rect(self.dataset_btn_pos, (150, 60))

        try:
            self.best_dqn_btn = pygame.image.load('images/best_dqn.png')
            self.best_dqn_btn = pygame.transform.scale(self.best_dqn_btn, (150, 60))
        except pygame.error as e:
            print(f"Error loading best_dqn_btn image: {e}")
            self.best_dqn_btn = None
        self.best_dqn_btn_pos = (WIDTH // 2 - (self.best_dqn_btn.get_width() // 2 if self.best_dqn_btn else 0), HEIGHT // 2 + 10)
        self.best_dqn_btn_rect = self.best_dqn_btn.get_rect(topleft=self.best_dqn_btn_pos) if self.best_dqn_btn else pygame.Rect(self.best_dqn_btn_pos, (150, 60))

        try:
            self.best_dataset_btn = pygame.image.load('images/best_dataset.png')
            self.best_dataset_btn = pygame.transform.scale(self.best_dataset_btn, (150, 60))
        except pygame.error as e:
            print(f"Error loading best_dataset_btn image: {e}")
            self.best_dataset_btn = None
        self.best_dataset_btn_pos = (WIDTH // 2 - (self.best_dataset_btn.get_width() // 2 if self.best_dataset_btn else 0), HEIGHT // 2 + 80)
        self.best_dataset_btn_rect = self.best_dataset_btn.get_rect(topleft=self.best_dataset_btn_pos) if self.best_dataset_btn else pygame.Rect(self.best_dataset_btn_pos, (150, 60))

        try:
            self.menu_btn = pygame.image.load('images/menu.png')
            self.menu_btn = pygame.transform.scale(self.menu_btn, (115, 40))
        except pygame.error as e:
            print(f"Error loading menu_btn image: {e}")
            self.menu_btn = None
        self.menu_btn_pos = (WIDTH - (self.menu_btn.get_width() if self.menu_btn else 0) - 10, 10)
        self.menu_btn_rect = self.menu_btn.get_rect(topleft=self.menu_btn_pos) if self.menu_btn else pygame.Rect(self.menu_btn_pos, (115, 40))

    def show_start(self):
        if self.logo:
            self.screen.blit(self.logo, self.logo_pos)
        if self.score_rect:
            self.screen.blit(self.score_rect, self.score_rect_pos)
        if self.best_rect:
            self.screen.blit(self.best_rect, self.best_rect_pos)
        if self.message:
            self.screen.blit(self.message, self.message_pos)
        if self.newgame_btn:
            self.screen.blit(self.newgame_btn, self.newgame_btn_pos)
        if self.ai_btn:
            self.screen.blit(self.ai_btn, self.ai_btn_pos)
        pygame.draw.rect(self.screen, BOARD_COLOR, self.board_rect)

    def update_scores(self, score_value, best_value):
        if self.score_rect:
            self.screen.blit(self.score_rect, self.score_rect_pos)
            self.score_lbl = self.score_font.render(str(score_value), 0, WHITE)
            self.score_pos = (self.score_rect_pos[0] + self.score_rect.get_width() // 2 - self.score_lbl.get_rect().width // 2,
                              self.score_rect_pos[1] + self.score_rect.get_height() // 2 - self.score_lbl.get_rect().height // 2 + 8)
            self.screen.blit(self.score_lbl, self.score_pos)

        if self.best_rect:
            self.screen.blit(self.best_rect, self.best_rect_pos)
            self.best_lbl = self.score_font.render(str(best_value), 0, WHITE)
            self.best_pos = (self.best_rect_pos[0] + self.best_rect.get_width() // 2 - self.best_lbl.get_rect().width // 2,
                             self.best_rect_pos[1] + self.best_rect.get_height() // 2 - self.best_lbl.get_rect().height // 2 + 8)
            self.screen.blit(self.best_lbl, self.best_pos)

    def action_listener(self, event):
        if self.menu.active:
            if self.menu.tryagain_btn_rect.collidepoint(event.pos):
                self.menu.hide(self.board_rect)
                return 'try_again'
        elif self.newgame_btn_rect and self.newgame_btn_rect.collidepoint(event.pos):
            return 'new_game'
        elif self.ai_btn_rect and self.ai_btn_rect.collidepoint(event.pos):
            return 'ai_toggle'
        elif self.menu_btn_rect and self.menu_btn_rect.collidepoint(event.pos):
            return 'menu'
        return False

    def initial_menu_listener(self, event):
        if self.dqn_btn_rect and self.dqn_btn_rect.collidepoint(event.pos):
            return 'dqn'
        elif self.dataset_btn_rect and self.dataset_btn_rect.collidepoint(event.pos):
            return 'dataset'
        elif self.best_dqn_btn_rect and self.best_dqn_btn_rect.collidepoint(event.pos):
            return 'best_dqn'
        elif self.best_dataset_btn_rect and self.best_dataset_btn_rect.collidepoint(event.pos):
            return 'best_dataset'
        return False

    def show_initial_menu(self):
        self.screen.fill(WHITE)
        if self.dqn_btn:
            self.screen.blit(self.dqn_btn, self.dqn_btn_pos)
        if self.dataset_btn:
            self.screen.blit(self.dataset_btn, self.dataset_btn_pos)
        if self.best_dqn_btn:
            self.screen.blit(self.best_dqn_btn, self.best_dqn_btn_pos)
        if self.best_dataset_btn:
            self.screen.blit(self.best_dataset_btn, self.best_dataset_btn_pos)

    def draw_board(self, tiles):
        pygame.draw.rect(self.screen, BOARD_COLOR, self.board_rect)
        rShift, cShift = GAP, GAP
        for row in range(ROWS):
            for col in range(COLS):
                tile_num = int(tiles[row][col])
                tile_color = TILES_COLORS[tile_num]
                rect = pygame.Rect(
                    self.board_rect[0] + cShift + col * TILE_SIZE,
                    self.board_rect[1] + rShift + row * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(self.screen, tile_color, rect)
                tile_lbl_color = LBLS_COLORS[tile_num]
                if tile_num > 0:
                    lbl = self.score_font.render(str(tile_num), 0, tile_lbl_color)
                    lbl_rect = lbl.get_rect(center=rect.center)
                    self.screen.blit(lbl, lbl_rect)
                cShift += GAP
            rShift += GAP
            cShift = GAP

    def show_controls(self):
        if self.newgame_btn:
            self.screen.blit(self.newgame_btn, self.newgame_btn_pos)
        if self.ai_btn:
            self.screen.blit(self.ai_btn, self.ai_btn_pos)

    def show_menu_button(self):
        if self.menu_btn:
            self.screen.blit(self.menu_btn, self.menu_btn_pos)

class Menu:
    def __init__(self, screen):
        self.screen = screen
        WIDTH, HEIGHT = screen.get_size()
        self.transparent_screen = pygame.Surface((WIDTH, HEIGHT))
        self.transparent_screen.set_alpha(TRANSPARENT_ALPHA)
        self.transparent_screen.fill(WHITE)

        self.go_font = pygame.font.SysFont('verdana', 30, True)
        self.go_lbl = self.go_font.render('Game over!', 1, GAMEOVER_LBL_COLOR)
        self.go_pos = (WIDTH // 2 - self.go_lbl.get_rect().width // 2, HEIGHT // 2 - self.go_lbl.get_rect().height // 2 - 35)

        try:
            self.tryagain_btn = pygame.image.load('images/tryagain_btn.png')
            self.tryagain_btn = pygame.transform.scale(self.tryagain_btn, (115, 40))
        except pygame.error as e:
            print(f"Error loading tryagain_btn image: {e}")
            self.tryagain_btn = None
        self.tryagain_btn_pos = (WIDTH // 2 - (self.tryagain_btn.get_width() // 2 if self.tryagain_btn else 0), HEIGHT // 2 - (self.tryagain_btn.get_height() // 2 if self.tryagain_btn else 0) + 35)
        self.tryagain_btn_rect = self.tryagain_btn.get_rect(topleft=self.tryagain_btn_pos) if self.tryagain_btn else pygame.Rect(self.tryagain_btn_pos, (115, 40))

        self.active = False

    def show(self):
        if self.active:
            self.screen.blit(self.transparent_screen, (0, 0))
            self.screen.blit(self.go_lbl, self.go_pos)
            if self.tryagain_btn:
                self.screen.blit(self.tryagain_btn, self.tryagain_btn_pos)

    def hide(self, bg):
        self.active = False
        pygame.draw.rect(self.screen, BOARD_COLOR, bg)
        pygame.display.update()
