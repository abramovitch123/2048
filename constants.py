
WIDTH, HEIGHT = 900, 850
BOARD_WIDTH, BOARD_HEIGHT = 400, 400
COLS, ROWS = 4, 4
GAP = 10
TILE_SIZE = (BOARD_WIDTH - (COLS + 1) * GAP) // COLS


WHITE = (255, 255, 255)
SCREEN_COLOR = (249, 246, 235)
BOARD_COLOR = (173, 157, 143)


GAMEOVER_LBL_COLOR = (119, 111, 102)
TRANSPARENT_ALPHA = 210

TILES_COLORS = {
    0: (194, 178, 166),
    2: (233, 221, 209),
    4: (232, 217, 189),
    8: (236, 161, 101),
    16: (241, 130, 80),
    32: (239, 100, 77),
    64: (240, 69, 45),
    128: (230, 197, 94),
    256: (227, 190, 78),
    512: (230, 189, 64),
    1024: (233, 185, 49),
    2048: (233, 187, 32),
    4096: (35, 32, 29),
    8192: (35, 32, 29),
}


AI_BTN_WIDTH, AI_BTN_HEIGHT = 115, 40
AI_BTN_X, AI_BTN_Y = WIDTH // 2 - AI_BTN_WIDTH // 2, HEIGHT - AI_BTN_HEIGHT - 60

LBLS_COLORS = {
    0: (194, 178, 166),
    2: (99, 91, 82),
    4: (99, 91, 82),
    8: WHITE,
    16: WHITE,
    32: WHITE,
    64: WHITE,
    128: WHITE,
    256: WHITE,
    512: WHITE,
    1024: WHITE,
    2048: WHITE,
    4096: WHITE,
    8192: WHITE,
}
