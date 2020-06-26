import tkinter as tk
import numpy as np
import gym
import random
from gym import spaces
from enum import Enum
from time import sleep

SCALE=30

class AppleCollector(gym.Env):
    def __init__(self):
        self.size = 17
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.size, self.size), dtype=np.uint8)
        self.ax, self.ay = 0, 0
        self.green_apples = []
        self.red_apples = []
        self.green_apple_good = True

        def up(self):
            self.ay = max(self.ay - 1, 0)
        def down(self):
            self.ay = min(self.ay + 1, self.size-1)
        def left(self):
            self.ax = max(self.ax - 1, 0)
        def right(self):
            self.ax = min(self.ax + 1, self.size-1)

        self.action_fx = {
            0 : up,
            1 : down,
            2 : left,
            3 : right,
        }
        self.episode_steps = 0
        self.max_episode_steps = 100
        self.render_initialized = False


    def seed(self, seed=None):
        pass

    def _spawn_apples(self, n):
        apples = []
        while len(apples) < n:
            x = np.random.randint(self.size)
            y = np.random.randint(self.size)
            if (x, y) not in apples:
                apples.append((x, y))
        return apples

    def reset(self):
        self.episode_steps = 0
        self.ax = np.random.randint(self.size)
        self.ay = np.random.randint(self.size)

        apples = self._spawn_apples(50)
        self.green_apples = apples[:len(apples)//2]
        self.red_apples = apples[len(apples)//2:]

        # self.green_apple_good = random.choice([True, False])

        return self._assemble_observation()

    def step(self, action):
        self.episode_steps += 1
        self.action_fx[action.item()](self)
        reward = 0
        if (self.ax, self.ay) in self.green_apples:
            reward += 1 if self.green_apple_good else -1
            self.green_apples.remove((self.ax, self.ay))
        if (self.ax, self.ay) in self.red_apples:
            reward -= 1 if self.green_apple_good else -1
            self.red_apples.remove((self.ax, self.ay))
        done = self.episode_steps >= self.max_episode_steps
        return self._assemble_observation(), reward, done, {}

    def _draw_oval(self, x, y, color="#fff"):
        self.canvas.create_oval((x + 0.5 - 0.2) * SCALE,
                                (y + 0.5 - 0.2) * SCALE,
                                (x + 0.5 + 0.2) * SCALE,
                                (y + 0.5 + 0.2) * SCALE,
                                outline=color, fill=color)

    def render(self, mode='human'):
        if not self.render_initialized:
            self._initialize_canvas()
            self.render_initialized = True
        agent_drawing_radius = 0.2
        self.canvas.delete('all')
        for x in range(self.size):
            for y in range(self.size):
                fill_color = "black"
                self.canvas.create_rectangle(y*SCALE, x*SCALE, (y+1)*SCALE, (x+1)*SCALE, outline="#fff", fill=fill_color)
        self._draw_oval(self.ax, self.ay)
        for x, y in self.green_apples:
            self._draw_oval(x, y, color="#00ff00")
        for x, y in self.red_apples:
            self._draw_oval(x, y, color="#ff0000")
        self.root.update()
        sleep(0.1)


    def _assemble_observation(self):
        obs = np.zeros([3, self.size, self.size], dtype=np.uint8)
        obs[0, self.ax, self.ay] = 1
        for x, y in self.green_apples:
            obs[1, x, y] = 1
        for x, y in self.red_apples:
            obs[2, x, y] = 1
        return obs

    def _initialize_canvas(self):
        self.root = tk.Tk()
        self.root.wm_title("AppleCollector")
        self.canvas = tk.Canvas(self.root, width=self.size*SCALE+1, height=self.size*SCALE+1, borderwidth=0, highlightthickness=0, bg="black")
        self.canvas.grid()
