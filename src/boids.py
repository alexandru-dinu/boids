#!/usr/bin/env python3

"""
Reynolds' boids
https://www.red3d.com/cwr/boids
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pygame
import yaml
from pygame.locals import QUIT


class Boid:
    def __init__(self, bid, pos, orientation, velocity, size, fov, cfgs):
        self.bid = bid

        self.v_pos = np.array([pos[0], pos[1]], dtype="float32")

        v = np.random.uniform(-velocity, velocity, size=(2,))
        self.v_vel = (v / np.linalg.norm(v)) * velocity
        self.v_acc = np.zeros(2)

        self.max_vel = velocity
        self.max_force = cfgs["boid_max_force"]
        self.size = size

        self.fov_angle, self.fov_radius = fov

        self.color = "#%06x" % np.random.randint(0xA0EF7F, 0xAAF756)

        self.cfgs = cfgs

    def __eq__(self, other):
        return self.bid == other.bid

    def __repr__(self):
        return "B<%3d>" % self.bid

    def __str__(self):
        return self.__repr__()

    @property
    def cx(self):
        return self.v_pos[0]

    @property
    def cy(self):
        return self.v_pos[1]

    @property
    def vx(self):
        return self.v_vel[0]

    @property
    def vy(self):
        return self.v_vel[1]

    @property
    def orientation(self):
        alpha = np.degrees(np.arctan2(self.vy, self.vx))
        if alpha < 0:
            alpha += 360
        return alpha

    @property
    def polygon(self):
        alpha = 140
        beta = 360 - 2 * alpha

        t = np.radians(self.orientation)
        a = (self.cx + self.size * np.cos(t), self.cy + self.size * np.sin(t))

        t += np.radians(alpha)
        b = (self.cx + self.size * np.cos(t), self.cy + self.size * np.sin(t))

        t += np.radians(beta)
        c = (self.cx + self.size * np.cos(t), self.cy + self.size * np.sin(t))

        return [a, b, c]

    def step(self, env, dt):
        # handle edges
        r = self.fov_radius

        dw0, dw1 = np.array([0, self.cy]), np.array([env.w, self.cy])
        dh0, dh1 = np.array([self.cx, 0]), np.array([self.cx, env.h])

        walls = [
            Obstacle((v[0], v[1]), radius=self.fov_radius // 2, is_wall=True)
            for v in [dw0, dw1, dh0, dh1]
        ]

        # handle obstacles
        self.v_acc += self.handle_obstacles(walls, dt)
        self.v_acc += self.handle_obstacles(env.obstacles, dt)

        # apply algorithm
        flockmates = self.get_flockmates(env.boids)

        self.v_acc += self.do_alignment(flockmates, dt)
        self.v_acc += self.do_cohesion(flockmates, dt)
        self.v_acc += self.do_separation(flockmates, dt)

        # update
        self.v_pos += self.v_vel
        self.v_vel += self.v_acc
        self.v_acc = np.zeros(2)

    def get_flockmates(self, others):
        """
        Flockmate = other boid who is within my FOV radius
        """
        return [
            o
            for o in others
            if o != self and np.linalg.norm(self.v_pos - o.v_pos) <= self.fov_radius
        ]

    def handle_obstacles(self, obstacles, dt):
        """
        Steer away from obstacles
        """
        avg = np.zeros(2)
        total = 0
        is_wall = True

        for obs in obstacles:
            is_wall = is_wall and obs.is_wall

            dist = np.linalg.norm(obs.v_pos - self.v_pos)
            if dist <= obs.radius + self.fov_radius:
                diff = (self.v_pos - obs.v_pos) / dist
                avg += diff
                total += 1

        if total == 0:
            return np.zeros(2)

        avg = avg / total

        if np.linalg.norm(avg) > 0:
            avg = (avg / np.linalg.norm(avg)) * self.max_vel

        v_steer = avg - self.v_vel

        f = self.cfgs["wall_factor"] if is_wall else 1.0

        return dt * f * self.cfgs["obstacle_factor"] * v_steer

    def do_alignment(self, flockmates, dt):
        """
        Steer towards the average orientation of the flockmates
        """
        n = len(flockmates)
        avg = np.zeros(2)

        if n == 0:
            return np.zeros(2)

        for other in flockmates:
            avg += other.v_vel / n

        avg = (avg / np.linalg.norm(avg)) * self.max_vel
        v_steer = avg - self.v_vel

        return dt * self.cfgs["align_factor"] * v_steer

    def do_cohesion(self, flockmates, dt):
        """
        Steer towards the mass centre of the flock
        """
        n = len(flockmates)
        avg = np.zeros(2)

        if n == 0:
            return np.zeros(2)

        for other in flockmates:
            avg += other.v_pos / n

        vec_to_com = avg - self.v_pos
        if np.linalg.norm(vec_to_com) > 0:
            vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_vel

        v_steer = vec_to_com - self.v_vel
        if np.linalg.norm(v_steer) > self.max_force:
            v_steer = (v_steer / np.linalg.norm(v_steer)) * self.max_force

        return dt * self.cfgs["cohesion_factor"] * v_steer

    def do_separation(self, flockmates, dt):
        """
        Steer away from flockmates (i.e. don't get too crowded)
        """
        n = len(flockmates)
        avg = np.zeros(2)

        if n == 0:
            return np.zeros(2)

        for other in flockmates:
            dist = np.linalg.norm(other.v_pos - self.v_pos)
            diff = (self.v_pos - other.v_pos) / dist
            avg += diff / n

        avg = (avg / np.linalg.norm(avg)) * self.max_vel
        v_steer = avg - self.v_vel

        if np.linalg.norm(v_steer) > self.max_force:
            v_steer = (v_steer / np.linalg.norm(v_steer)) * self.max_force

        return dt * self.cfgs["separation_factor"] * v_steer


# --- Boid


class Obstacle:
    def __init__(self, pos, radius, is_wall=False):
        self.v_pos = np.array([pos[0], pos[1]], dtype="float32")
        self.radius = radius
        self.color = "#eb6782"
        self.is_wall = is_wall

    @property
    def center(self):
        return self.v_pos


# --- Obstacle


class Environment:
    def __init__(self, dims, boids, obstacles):
        self.w, self.h = dims
        self.boids = boids
        self.obstacles = obstacles


# --- Environment


def setup_boids(cfg):
    boids = []

    for i in range(cfg["initial_boids"]):
        r = cfg["fov_radius"] + cfg["boid_size"]

        x = np.random.randint(r, cfg["env_width"] - r)
        y = np.random.randint(r, cfg["env_height"] - r)
        orientation = np.random.randint(0, 360)

        fov = (cfg["fov_angle"], cfg["fov_radius"])

        boid = Boid(
            i,
            (x, y),
            orientation,
            cfg["boid_velocity"],
            cfg["boid_size"],
            fov,
            cfgs=cfg["algorithm_cfgs"],
        )
        boids.append(boid)

    return boids


def main():
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    # set-up boids and environment
    boids = setup_boids(cfg)
    env = Environment((cfg["env_width"], cfg["env_height"]), boids, obstacles=[])

    # set-up pygame
    pygame.init()
    pygame.display.set_caption("Reynolds' boids")
    delta_t = 1000 // cfg["fps"]
    canvas = pygame.display.set_mode(
        size=(cfg["env_width"], cfg["env_height"]), flags=0, depth=32
    )

    t0 = 0

    # main loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit(0)

            if event.type == pygame.MOUSEBUTTONUP:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                obstacle = Obstacle((mouse_x, mouse_y), radius=cfg["obstacle_radius"])
                env.obstacles.append(obstacle)
        # --- events

        canvas.fill(pygame.Color("Black"))

        t1 = pygame.time.get_ticks()
        dt = (t1 - t0) / 1000.0  # delta time in seconds
        t0 = t1

        for b in boids:
            b.step(env, dt)
            pygame.draw.polygon(canvas, pygame.Color(b.color), b.polygon)
            if cfg["draw_fov"]:
                pygame.draw.circle(
                    canvas,
                    pygame.Color(b.color),
                    (int(b.cx), int(b.cy)),
                    int(b.fov_radius),
                    1,
                )
                pygame.draw.circle(
                    canvas,
                    pygame.Color(b.color),
                    (int(b.cx), int(b.cy)),
                    int(b.size),
                    1,
                )

        for o in env.obstacles:
            if not o.is_wall:
                pygame.draw.circle(canvas, pygame.Color(o.color), o.center, o.radius)

        pygame.display.update()
        pygame.time.delay(delta_t)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=Path, required=True, help="Boids configuration."
    )
    args = parser.parse_args()
    main()
