import pygame
import random
from enum import Enum
from collections import deque, namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
small_font = pygame.font.Font('arial.ttf', 15)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (250, 248, 232)
CREAM = (246, 226, 175)
GRASS_LIGHT = (74, 139, 71)
GRASS_DARK = (60, 124, 65)
GRASS_DETAIL = (85, 151, 76)
WOOD_DARK = (82, 51, 31)
WOOD_LIGHT = (147, 99, 57)
PANEL = (25, 51, 35)
PANEL_BORDER = (150, 186, 104)
SNAKE_DARK = (35, 96, 53)
SNAKE_BODY = (82, 176, 77)
SNAKE_LIGHT = (153, 218, 102)
EYE_WHITE = (250, 248, 224)
EYE_BLACK = (21, 27, 19)
APPLE_RED = (205, 43, 48)
APPLE_LIGHT = (249, 108, 91)
APPLE_DARK = (146, 30, 38)
LEAF = (56, 130, 61)
STEM = (91, 57, 31)
SHADOW = (42, 88, 48)

BLOCK_SIZE = 20
SPEED = 40
LOOP_HISTORY = 80
LOOP_REPEAT_LIMIT = 4

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.recent_states = deque(maxlen=LOOP_HISTORY)


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        state_key = (self.head, self.direction, self.food)
        repeated_state = self.recent_states.count(state_key) >= LOOP_REPEAT_LIMIT
        if self.is_collision() or self.frame_iteration > 100*len(self.snake) or repeated_state:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.recent_states.clear()
        else:
            self.snake.pop()

        self.recent_states.append(state_key)
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def select_safe_action(self, action):
        action_scores = [
            self._reachable_space([1, 0, 0]),
            self._reachable_space([0, 1, 0]),
            self._reachable_space([0, 0, 1]),
        ]
        selected_index = self._action_index(action)
        best_index = int(np.argmax(action_scores))
        selected_score = action_scores[selected_index]
        best_score = action_scores[best_index]
        minimum_escape_space = max(len(self.snake) + 2, 12)

        if selected_score == 0 or (
            selected_score < minimum_escape_space
            and best_score > selected_score
        ):
            safe_action = [0, 0, 0]
            safe_action[best_index] = 1
            return safe_action

        return action

    def _reachable_space(self, action):
        next_head = self._next_head(self._direction_for_action(action))
        # On a normal move the tail vacates its cell, making it a valid exit.
        blocked = set(self.snake[:-1] if next_head != self.food else self.snake)
        if (
            next_head.x < 0
            or next_head.x > self.w - BLOCK_SIZE
            or next_head.y < 0
            or next_head.y > self.h - BLOCK_SIZE
            or next_head in blocked
        ):
            return 0

        blocked.discard(next_head)
        visited = {next_head}
        pending = deque([next_head])

        while pending:
            point = pending.popleft()
            neighbors = (
                Point(point.x + BLOCK_SIZE, point.y),
                Point(point.x - BLOCK_SIZE, point.y),
                Point(point.x, point.y + BLOCK_SIZE),
                Point(point.x, point.y - BLOCK_SIZE),
            )
            for neighbor in neighbors:
                if neighbor in visited or neighbor in blocked:
                    continue
                if (
                    neighbor.x < 0
                    or neighbor.x > self.w - BLOCK_SIZE
                    or neighbor.y < 0
                    or neighbor.y > self.h - BLOCK_SIZE
                ):
                    continue
                visited.add(neighbor)
                pending.append(neighbor)

        return len(visited)

    def _action_index(self, action):
        if np.array_equal(action, [1, 0, 0]):
            return 0
        if np.array_equal(action, [0, 1, 0]):
            return 1
        return 2

    def _direction_for_action(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            return clock_wise[idx]
        if np.array_equal(action, [0, 1, 0]):
            return clock_wise[(idx + 1) % 4]
        return clock_wise[(idx - 1) % 4]

    def _next_head(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        return Point(x, y)


    def _update_ui(self):
        self._draw_background()
        self._draw_food()
        self._draw_snake()
        self._draw_score()
        pygame.display.flip()

    def _draw_background(self):
        for y in range(0, self.h, BLOCK_SIZE):
            for x in range(0, self.w, BLOCK_SIZE):
                color = GRASS_LIGHT if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else GRASS_DARK
                pygame.draw.rect(self.display, color, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))

                if (x // BLOCK_SIZE * 3 + y // BLOCK_SIZE) % 7 == 0:
                    pygame.draw.line(self.display, GRASS_DETAIL, (x + 5, y + 15), (x + 7, y + 11), 1)
                    pygame.draw.line(self.display, GRASS_DETAIL, (x + 7, y + 15), (x + 10, y + 12), 1)

        pygame.draw.rect(self.display, WOOD_DARK, pygame.Rect(0, 0, self.w, self.h), 6)
        pygame.draw.rect(self.display, WOOD_LIGHT, pygame.Rect(6, 6, self.w - 12, self.h - 12), 2)

    def _draw_food(self):
        center_x = int(self.food.x + BLOCK_SIZE // 2)
        center_y = int(self.food.y + BLOCK_SIZE // 2 + 1)

        pygame.draw.circle(self.display, SHADOW, (center_x + 2, center_y + 3), 9)
        pygame.draw.circle(self.display, APPLE_DARK, (center_x, center_y + 1), 9)
        pygame.draw.circle(self.display, APPLE_RED, (center_x, center_y), 8)
        pygame.draw.circle(self.display, APPLE_LIGHT, (center_x - 3, center_y - 3), 3)
        pygame.draw.line(self.display, STEM, (center_x, center_y - 7), (center_x + 2, center_y - 13), 3)
        pygame.draw.ellipse(self.display, LEAF, pygame.Rect(center_x + 1, center_y - 14, 9, 5))

    def _draw_snake(self):
        centers = [
            (int(pt.x + BLOCK_SIZE // 2), int(pt.y + BLOCK_SIZE // 2))
            for pt in self.snake
        ]

        for index in range(len(centers) - 1, 0, -1):
            center = centers[index]
            next_center = centers[index - 1]
            radius = 7 if index == len(centers) - 1 else 8
            pygame.draw.line(self.display, SNAKE_DARK, center, next_center, radius * 2)
            pygame.draw.circle(self.display, SNAKE_BODY, center, radius)
            pygame.draw.circle(self.display, SNAKE_LIGHT, (center[0] - 2, center[1] - 2), 3)

        head_center = centers[0]
        pygame.draw.circle(self.display, SNAKE_DARK, (head_center[0] + 1, head_center[1] + 2), 11)
        pygame.draw.circle(self.display, SNAKE_BODY, head_center, 10)
        pygame.draw.circle(self.display, SNAKE_LIGHT, (head_center[0] - 3, head_center[1] - 3), 4)
        self._draw_snake_face(head_center)

    def _draw_snake_face(self, center):
        eye_offsets = {
            Direction.RIGHT: ((4, -4), (4, 4)),
            Direction.LEFT: ((-4, -4), (-4, 4)),
            Direction.UP: ((-4, -4), (4, -4)),
            Direction.DOWN: ((-4, 4), (4, 4)),
        }
        pupil_offset = {
            Direction.RIGHT: (1, 0),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
        }

        pupil_x, pupil_y = pupil_offset[self.direction]
        for offset_x, offset_y in eye_offsets[self.direction]:
            eye_center = (center[0] + offset_x, center[1] + offset_y)
            pygame.draw.circle(self.display, EYE_WHITE, eye_center, 3)
            pygame.draw.circle(self.display, EYE_BLACK, (eye_center[0] + pupil_x, eye_center[1] + pupil_y), 1)

    def _draw_score(self):
        panel = pygame.Surface((170, 52), pygame.SRCALPHA)
        pygame.draw.rect(panel, (*PANEL, 225), pygame.Rect(0, 0, 170, 52), border_radius=9)
        pygame.draw.rect(panel, (*PANEL_BORDER, 235), pygame.Rect(0, 0, 170, 52), 1, border_radius=9)
        self.display.blit(panel, (14, 14))

        score_text = font.render("Score: " + str(self.score), True, WHITE)
        mode_text = small_font.render("AI TRAINING MODE", True, CREAM)
        self.display.blit(score_text, (25, 16))
        self.display.blit(mode_text, (26, 42))


    def _move(self, action):
        # [straight, right, left]

        self.direction = self._direction_for_action(action)
        self.head = self._next_head(self.direction)
