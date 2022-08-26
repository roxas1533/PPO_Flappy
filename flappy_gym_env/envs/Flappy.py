import os

import gym
import numpy as np
import pygame
from PIL import Image
from pygame.locals import *

from . import config

WIDTH = 400
HEIGHT = 400
pachage_dir = os.path.dirname(__file__)

flappyBird = pygame.image.load(f"{pachage_dir}/Flappy.png")
pipe = pygame.image.load(f"{pachage_dir}/pipe-green.png")


class Box:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.veloX = 0
        self.veloY = 0
        self.width = width
        self.height = height
        self.isDeath = False
        self.tag = "None"

    def draw(self, pygame, screen):
        pass

    def update(self):
        self.x += self.veloX
        self.y += self.veloY

    def col(self, obj2):
        if (
            self.x < obj2.x + obj2.width
            and self.x + self.width > obj2.x
            and self.y < obj2.y + obj2.height
            and self.y + self.height > obj2.y
        ):
            return True


class Player(Box):
    G = 1

    def __init__(self, x, y, width, height):
        global flappyBird
        super().__init__(x, y, width, height)
        self.time = 0
        self.reward = []
        self.history = []
        self.historyY = []
        flappyBird = pygame.transform.scale(flappyBird, (self.width, self.width))

    def draw(self, pygame, screen):
        screen.blit(flappyBird, (self.x, self.y))

    def update(self):
        self.time += 1
        super().update()
        self.veloY += self.G
        self.veloY = max(self.veloY, -8)
        self.veloY = min(self.veloY, 10)
        if self.y < 0 or self.y + self.height > HEIGHT:
            self.isDeath = True

    def jump(self):
        self.veloY = -9

    def do(self, X):
        type = np.random.choice(2)
        if type == 1:
            self.jump()
        return type

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time


class Object(Box):
    def __init__(self, x, y, width, height, tag):
        global pipe
        super().__init__(x, y, width, height)
        self.veloX = -4
        self.tag = "OUT"
        self.tag2 = tag
        t = pipe.get_width() / width
        self.pipe2 = pipe
        if tag == "UP":
            self.pipe2 = pygame.transform.flip(self.pipe2, False, True)

    def draw(self, pygame, screen):
        if self.tag2 == "DOWN":
            screen.blit(
                self.pipe2,
                (self.x, self.y),
                (0, 0, self.pipe2.get_width(), self.height),
            )
        else:
            screen.blit(
                self.pipe2,
                (self.x, self.y),
                (
                    0,
                    self.pipe2.get_height() - self.height,
                    self.pipe2.get_width(),
                    self.height,
                ),
            )

    def update(self):
        super().update()
        if self.x + self.width < 0:
            self.isDeath = True


class NextPoint(Box):
    def draw(self, pygame, screen):
        pass


class Dummy(Box):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.veloX = -4
        self.tag = "OK"
        self.tag2 = "no"

    def draw(self, pygame, screen):
        pass


class FlappyClass(gym.Env):
    def __init__(self):
        if not config.render_display:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"

        self.done = True
        self.player = Player(100, HEIGHT / 2, 25, 25)
        self.objects = []
        self.min = 1000
        self.time = 0
        self.isInited = False
        self.screen = None
        self.clock = None
        self.point = NextPoint(0, 0, 5, 5)
        self.action_space = gym.spaces.Discrete(2)
        self.reward = 0

        self.finish = False
        self.render()
        if config.train_type == "mlp":
            self.observation_space = gym.spaces.Box(
                low=0, high=400, shape=(5,), dtype=np.float32
            )
        elif config.train_type == "cnn":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(256, 256, 1), dtype=np.uint8
            )
        else:
            raise ValueError("train_type must be mlp or cnn")

    def render(self, mode="human"):
        if not self.isInited:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("GA_FLAPPY")
            self.clock = pygame.time.Clock()
            self.isInited = True
        for event in pygame.event.get():  # 終了処理
            if event.type == KEYDOWN:
                self.player.jump()
            if event.type == QUIT:
                self.finish = True
                break
        if not self.finish:
            self.screen.fill((0, 0, 0))

        for o in self.objects[:]:
            o.draw(pygame=pygame, screen=self.screen)
        self.player.draw(pygame=pygame, screen=self.screen)
        self.point.draw(pygame=pygame, screen=self.screen)
        pygame.display.update()

    def step(self, action):
        done = False
        pointFlag = False
        self.reward = 0
        if action == 1:
            self.player.jump()
        state_data = []
        for o in self.objects[:]:
            o.update()
            if o.tag == "OK":
                state_data.append(o.x)
                state_data.append(o.y)

            if 0 < (o.x + o.width - self.player.x) and not pointFlag and o.tag2 == "UP":
                self.min = o.x + o.width - self.player.x
                self.point.x = o.x + o.width
                self.point.y = o.y + o.height + 60
                pointFlag = True
            if o.col(self.player):
                if o.tag == "OK":
                    o.isDeath = True
                    self.reward += 0.8
                else:
                    self.player.isDeath = True
            if o.isDeath:
                self.objects.remove(o)

        if len(state_data) == 2:
            state_data.append(0)
            state_data.append(0)
        if not self.player.isDeath:
            self.player.update()
            self.reward += 0.1

        else:
            self.reward = -0.8
            done = True

        if self.time % 50 == 0:
            rand = 0
            while rand == 0:
                rand = int(np.random.rand() * (HEIGHT - 150))
            self.objects.append(Object(WIDTH, 0, 40, rand, tag="UP"))
            self.objects.append(Object(WIDTH, rand + 120, 40, HEIGHT, tag="DOWN"))
            self.objects.append(Dummy(WIDTH + 25 + 15, rand, 5, 120))
        self.time += 1
        if not self.finish:
            self.render()
        if config.render_display:
            self.clock.tick(60)

        return (
            self.WriteState(state_data),
            self.reward,
            done,
            {"finish": self.finish},
        )

    def reset(self):
        self.player = Player(100, HEIGHT / 2, 25, 25)
        self.point = NextPoint(0, 0, 5, 5)
        self.objects = []
        self.time = 0
        self.min = 1000
        self.isInited = False
        self.screen = None
        self.clock = None

        self.reward = 0
        rand = int(np.random.rand() * (HEIGHT - 150))
        self.render()

        return self.WriteState()

    def WriteState(self, state_data=[0, 0, 0, 0]):
        if config.train_type == "cnn":
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            pilImg = Image.fromarray(np.uint8(image_data))
            del image_data
            pilImg = pilImg.convert("L")
            pilImg = pilImg.resize((256, 256))
            pilImg = pilImg.rotate(-90)
            ImgArray = np.asarray(pilImg)
            state = ImgArray[:, :, np.newaxis]
        elif config.train_type == "mlp":
            if len(state_data) == 0:
                state_data = [0, 0, 0, 0]
            state = [self.player.y + 25 / 2] + state_data
            state = np.array(state) / 400
        else:
            raise ValueError("train_type must be either mlp or cnn")
        return state


if __name__ == "__main__":
    flappy = FlappyClass()
    flappy.reset()
    while True:

        _, _, done, f = flappy.step(0)
        if f["finish"]:
            break
        flappy.render()
        if done:
            flappy.reset()
