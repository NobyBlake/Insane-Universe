#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import sys
import random


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'mass')

    def __init__(self, x, y, vx=0.0, vy=0.0, mass=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass


class World:
    def __init__(self, width=1000.0, height=1000.0, boundary='torus'):
        self.width = width
        self.height = height
        self.boundary = boundary

    def is_inside(self, x, y):
        return (0.0 <= x <= self.width) and (0.0 <= y <= self.height)

    def apply_boundary(self, particles):
        if self.boundary == 'torus':
            for p in particles:
                p.x = p.x % self.width
                p.y = p.y % self.height
        elif self.boundary == 'reflect':
            for p in particles:
                if p.x < 0.0:
                    p.x = -p.x
                    p.vx = -p.vx
                elif p.x > self.width:
                    p.x = 2.0 * self.width - p.x
                    p.vx = -p.vx
                if p.y < 0.0:
                    p.y = -p.y
                    p.vy = -p.vy
                elif p.y > self.height:
                    p.y = 2.0 * self.height - p.y
                    p.vy = -p.vy
        elif self.boundary == 'open':
            pass

    def get_minimum_image(self, dx, dy):
        if self.boundary == 'torus':
            dx = dx - self.width * round(dx / self.width)
            dy = dy - self.height * round(dy / self.height)
        return dx, dy


class Physics:
    def __init__(self, formula_idx=0, a=1.0, b=0.0, c=1.0, softening=0.5, antimatter_interaction=True):
        self.formula_idx = formula_idx
        self.a = a
        self.b = b
        self.c = c
        self.softening = softening
        self.antimatter_interaction = antimatter_interaction

    def compute_force(self, p1, p2, world):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dx, dy = world.get_minimum_image(dx, dy)

        dist2 = dx * dx + dy * dy + self.softening * self.softening
        dist = math.sqrt(dist2)
        if dist < 0.0001:
            return 0.0, 0.0

        m1, m2 = p1.mass, p2.mass
        coeff = self.a * (10.0 ** self.b)
        mass_product = m1 * m2

        if self.formula_idx == 0:
            f = coeff * mass_product / dist
        elif self.formula_idx == 1:
            f = coeff * mass_product / dist2
        elif self.formula_idx == 2:
            f = -coeff * dist
        elif self.formula_idx == 3:
            f = -coeff * mass_product * dist
        elif self.formula_idx == 4:
            f = coeff * mass_product * math.exp(-dist / self.c) / dist
        elif self.formula_idx == 5:
            f = coeff * mass_product * math.exp(-dist / self.c) / dist2
        else:
            f = 0.0

        # Если галочка выключена и знаки разные — обнуляем силу (кроме формулы 2, где mass_product не участвует)
        if not self.antimatter_interaction and mass_product < 0:
            if self.formula_idx != 2:
                f = 0.0

        return f * dx / dist, f * dy / dist

    def compute_accelerations(self, particles, world):
        n = len(particles)
        ax, ay = np.zeros(n), np.zeros(n)
        for i in range(n):
            fx_total = fy_total = 0.0
            for j in range(n):
                if i == j:
                    continue
                fx, fy = self.compute_force(particles[i], particles[j], world)
                fx_total += fx
                fy_total += fy
            m = particles[i].mass
            if m != 0.0:
                ax[i] = fx_total / abs(m)
                ay[i] = fy_total / abs(m)
        return ax, ay

    def euler_step(self, particles, world, dt):
        ax, ay = self.compute_accelerations(particles, world)
        for i, p in enumerate(particles):
            p.vx += ax[i] * dt
            p.vy += ay[i] * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
        return particles


class Merger:
    def __init__(self, enabled=True, density=1.0, antimatter_interaction=True):
        self.enabled = enabled
        self.density = density
        self.antimatter_interaction = antimatter_interaction

    def get_radius(self, mass):
        abs_mass = abs(mass)
        if abs_mass <= 0:
            return 0.1
        return math.sqrt(abs_mass / (math.pi * self.density))

    def find_pairs(self, particles, world):
        pairs = set()
        n = len(particles)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = particles[i], particles[j]
                dx, dy = world.get_minimum_image(p1.x - p2.x, p1.y - p2.y)
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.get_radius(p1.mass) + self.get_radius(p2.mass):
                    pairs.add((i, j))
        return pairs

    def merge(self, particles, world):
        if not self.enabled or len(particles) < 2:
            return particles

        pairs = self.find_pairs(particles, world)
        if not pairs:
            return particles

        n = len(particles)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i, j in pairs:
            union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        new_particles = []
        for indices in groups.values():
            if len(indices) == 1:
                new_particles.append(particles[indices[0]])
                continue

            # Проверяем, есть ли в группе и положительные, и отрицательные массы
            has_positive = any(particles[idx].mass > 0 for idx in indices)
            has_negative = any(particles[idx].mass < 0 for idx in indices)

            # Если галочка выключена и есть разные знаки — не сливаем, оставляем как есть
            if not self.antimatter_interaction and has_positive and has_negative:
                for idx in indices:
                    new_particles.append(particles[idx])
                continue

            # Иначе сливаем (арифметическое сложение масс)
            total_mass = 0.0
            cx = cy = px = py = 0.0
            for idx in indices:
                p = particles[idx]
                total_mass += p.mass
                cx += p.x * p.mass
                cy += p.y * p.mass
                px += p.vx * p.mass
                py += p.vy * p.mass

            if total_mass != 0.0:
                new_particles.append(Particle(
                    cx / total_mass, cy / total_mass,
                    px / total_mass, py / total_mass,
                    total_mass
                ))

        return new_particles


class RandomGenerator:
    def __init__(self, particles_per_100=0):
        self.particles_per_100 = particles_per_100
        self.schedule = []
        self.current_step_in_window = 0

    def set_params(self, particles_per_100):
        self.particles_per_100 = particles_per_100
        self.reset_schedule()

    def reset_schedule(self):
        self.schedule = []
        self.current_step_in_window = 0
        if self.particles_per_100 > 0:
            self.schedule = [random.randint(1, 100) for _ in range(self.particles_per_100)]

    def get_particles_for_step(self):
        if self.particles_per_100 == 0:
            return 0

        self.current_step_in_window += 1

        if self.current_step_in_window > 100:
            self.current_step_in_window = 1
            self.schedule = [random.randint(1, 100) for _ in range(self.particles_per_100)]

        return self.schedule.count(self.current_step_in_window)


class Renderer:
    def __init__(self, canvas, world, density=1.0):
        self.canvas = canvas
        self.world = world
        self.density = density
        self.min_radius = 1.0

    def get_radius(self, mass):
        abs_mass = abs(mass)
        if abs_mass <= 0:
            return self.min_radius
        return math.sqrt(abs_mass / (math.pi * self.density))

    def mass_to_color(self, mass):
        abs_mass = abs(mass)
        if abs_mass <= 1.0:
            t = 0.0
        elif abs_mass >= 100.0:
            t = 1.0
        else:
            t = (abs_mass - 1.0) / 99.0

        if mass > 0:
            r1, g1, b1 = 255, 228, 181
            r2, g2, b2 = 139, 69, 19
        else:
            r1, g1, b1 = 173, 216, 230
            r2, g2, b2 = 19, 48, 129

        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def draw(self, particles):
        self.canvas.delete("all")

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 800, 600

        scale = min(canvas_w / self.world.width, canvas_h / self.world.height)
        display_w = self.world.width * scale
        display_h = self.world.height * scale
        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2

        self.canvas.config(bg="#3e3e3e")
        self.canvas.create_rectangle(
            offset_x, offset_y,
            offset_x + display_w, offset_y + display_h,
            fill="#000000", outline=""
        )

        if not particles:
            return

        for p in particles:
            screen_x = offset_x + p.x * scale
            screen_y = offset_y + p.y * scale
            radius = max(self.get_radius(p.mass) * scale, self.min_radius)
            self.canvas.create_oval(
                screen_x - radius, screen_y - radius,
                screen_x + radius, screen_y + radius,
                fill=self.mass_to_color(p.mass), outline=""
            )


class Simulation:
    def __init__(self, world, physics, merger, random_gen, renderer):
        self.world = world
        self.physics = physics
        self.merger = merger
        self.random_gen = random_gen
        self.renderer = renderer
        self.particles = []
        self.running = False
        self.step_counter = 0
        self.dt = 0.1

    def reset(self):
        self.particles = []
        self.step_counter = 0
        self.running = False
        self.random_gen.reset_schedule()

    def add_particle_pair(self, mass=1.0):
        x = random.uniform(0.0, self.world.width)
        y = random.uniform(0.0, self.world.height)

        angle = random.uniform(0.0, 2.0 * math.pi)
        distance = random.uniform(1.0, 3.0)

        x1 = x + distance * math.cos(angle)
        y1 = y + distance * math.sin(angle)
        x2 = x - distance * math.cos(angle)
        y2 = y - distance * math.sin(angle)

        x1 = max(0.0, min(self.world.width, x1))
        y1 = max(0.0, min(self.world.height, y1))
        x2 = max(0.0, min(self.world.width, x2))
        y2 = max(0.0, min(self.world.height, y2))

        self.particles.append(Particle(x1, y1, mass=mass))
        self.particles.append(Particle(x2, y2, mass=-mass))

    def add_particles(self, count, mass=1.0):
        for _ in range(count):
            self.add_particle_pair(mass)

    def update_step(self):
        if not self.running:
            return

        self.physics.euler_step(self.particles, self.world, self.dt)

        if self.world.boundary == 'open':
            self.particles = [p for p in self.particles if self.world.is_inside(p.x, p.y)]
        else:
            self.world.apply_boundary(self.particles)

        self.particles = self.merger.merge(self.particles, self.world)

        count = self.random_gen.get_particles_for_step()
        if count > 0:
            self.add_particles(count, mass=1.0)

        self.step_counter += 1

    def get_stats(self):
        return {'count': len(self.particles), 'steps': self.step_counter}

    def get_top_masses(self, n=15):
        sorted_particles = sorted(self.particles, key=lambda p: abs(p.mass), reverse=True)
        return [int(p.mass) for p in sorted_particles[:n]]


class UniverseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Insane Universe. Antimatter")

        if sys.platform == "win32":
            self.root.state("zoomed")
        else:
            self.root.attributes("-zoomed", True)

        self.root.update_idletasks()
        screen_w = max(self.root.winfo_width(), 100)
        screen_h = max(self.root.winfo_height(), 100)

        if screen_w <= 100:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()

        panel_width = int(screen_w * 0.15)
        canvas_width = screen_w - panel_width
        canvas_height = screen_h

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(main_frame, width=canvas_width, height=canvas_height)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#3e3e3e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.panel = ttk.Notebook(main_frame, width=panel_width)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.world = World()
        self.physics = Physics()
        self.merger = Merger(antimatter_interaction=True)
        self.random_gen = RandomGenerator()
        self.renderer = Renderer(self.canvas, self.world)
        self.sim = Simulation(self.world, self.physics, self.merger, self.random_gen, self.renderer)

        self.create_sim_tab()
        self.create_settings_tab()
        self.create_random_tab()

        self.sim.reset()
        self.update_stats()
        self.renderer.draw(self.sim.particles)

        self.after_id = None
        self.steps_per_frame = 1

        self.canvas.bind("<Configure>", lambda e: self.renderer.draw(self.sim.particles))

    def create_sim_tab(self):
        tab = tk.Frame(self.panel, bg="#1e1e1e")
        self.panel.add(tab, text="Симуляция")

        c = tk.Frame(tab, bg="#1e1e1e")
        c.pack(fill=tk.Y, expand=True, padx=10, pady=10)

        tk.Label(c, text="Добавить пары", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0))

        add_frame = tk.Frame(c, bg="#1e1e1e")
        add_frame.pack(pady=5, fill=tk.X)

        self.add_slider = tk.Scale(add_frame, from_=1, to=1000, resolution=1,
                                   orient=tk.HORIZONTAL, showvalue=0,
                                   bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.add_slider.set(100)
        self.add_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.add_label = tk.Label(add_frame, text="100", fg="white", bg="#1e1e1e", width=5)
        self.add_label.pack(side=tk.LEFT)
        self.add_slider.config(command=lambda v: self.add_label.config(text=str(int(float(v)))))

        tk.Button(c, text="Добавить", command=self.add_particles,
                  width=18, bg="#2a5a2a", fg="white").pack(pady=5)

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        self.count_label = tk.Label(c, text="Точек: 0", fg="#88ff88", bg="#1e1e1e",
                                    font=("Arial", 10, "bold"), anchor="w")
        self.count_label.pack(fill=tk.X, pady=1)

        self.step_label = tk.Label(c, text="Шаг: 0", fg="#ffff88", bg="#1e1e1e",
                                   font=("Arial", 10, "bold"), anchor="w")
        self.step_label.pack(fill=tk.X, pady=1)

        tk.Label(c, text="Топ-15 масс", fg="white", bg="#1e1e1e",
                 font=("Arial", 10)).pack(pady=(10, 0))

        self.top_labels = []
        for i in range(15):
            lbl = tk.Label(c, text=f"{i+1}. ---", fg="#88ddff", bg="#1e1e1e",
                           font=("Courier", 9), anchor="w")
            lbl.pack(pady=1, fill=tk.X)
            self.top_labels.append(lbl)

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Label(c, text="Скорость (шагов за кадр)", fg="white", bg="#1e1e1e",
                 font=("Arial", 10)).pack(pady=(5, 0))

        speed_frame = tk.Frame(c, bg="#1e1e1e")
        speed_frame.pack(pady=5, fill=tk.X)

        self.speed_slider = tk.Scale(speed_frame, from_=1, to=10, resolution=1,
                                     orient=tk.HORIZONTAL, showvalue=0,
                                     command=self.set_speed,
                                     bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.speed_slider.set(1)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.speed_label = tk.Label(speed_frame, text="1", fg="white", bg="#1e1e1e", width=5)
        self.speed_label.pack(side=tk.LEFT)

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        self.start_stop_btn = tk.Button(c, text="Старт", command=self.toggle_start_stop,
                                        width=18, bg="#3e3e3e", fg="white")
        self.start_stop_btn.pack(pady=(0, 5))

        self.reset_btn = tk.Button(c, text="Сброс", command=self.reset_sim,
                                   width=18, bg="#3e3e3e", fg="white")
        self.reset_btn.pack(pady=(0, 0))

        tk.Frame(c, height=1, bg="#1e1e1e").pack(fill=tk.Y, expand=True)

    def create_settings_tab(self):
        tab = tk.Frame(self.panel, bg="#1e1e1e")
        self.panel.add(tab, text="Настройки")

        c = tk.Frame(tab, bg="#1e1e1e")
        c.pack(fill=tk.Y, expand=True, padx=10, pady=10)

        tk.Label(c, text="Размеры мира", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        l_frame = tk.Frame(c, bg="#1e1e1e")
        l_frame.pack(fill=tk.X, pady=2)
        tk.Label(l_frame, text="L", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.l_slider = tk.Scale(l_frame, from_=10, to=3000, resolution=10,
                                 orient=tk.HORIZONTAL, showvalue=0,
                                 bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.l_slider.set(1000)
        self.l_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.l_label = tk.Label(l_frame, text="1000", fg="white", bg="#1e1e1e", width=6)
        self.l_label.pack(side=tk.LEFT, padx=5)
        self.l_slider.config(command=lambda v: self.l_label.config(text=str(int(float(v)))))

        h_frame = tk.Frame(c, bg="#1e1e1e")
        h_frame.pack(fill=tk.X, pady=2)
        tk.Label(h_frame, text="H", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.h_slider = tk.Scale(h_frame, from_=10, to=3000, resolution=10,
                                 orient=tk.HORIZONTAL, showvalue=0,
                                 bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.h_slider.set(1000)
        self.h_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.h_label = tk.Label(h_frame, text="1000", fg="white", bg="#1e1e1e", width=6)
        self.h_label.pack(side=tk.LEFT, padx=5)
        self.h_slider.config(command=lambda v: self.h_label.config(text=str(int(float(v)))))

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Label(c, text="Логика границ", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        self.boundary_var = tk.StringVar(value="torus")
        for text, val in [("Тор", "torus"), ("Плоскость (отражение)", "reflect"), ("Плоскость (вылет)", "open")]:
            tk.Radiobutton(c, text=text, variable=self.boundary_var, value=val,
                           bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W)

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Label(c, text="Слияние частиц", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        self.merge_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Включено (по радиусам)", variable=self.merge_var,
                       bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W)

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Label(c, text="Формула силы", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        self.formula_var = tk.IntVar(value=0)
        formulas = [
            "a*10^b * m1*m2 / r",
            "a*10^b * m1*m2 / r^2",
            "-a*10^b * r",
            "-a*10^b * m1*m2 * r",
            "a*10^b * m1*m2 * e^(-r/c) / r",
            "a*10^b * m1*m2 * e^(-r/c) / r^2",
        ]
        for i, txt in enumerate(formulas):
            tk.Radiobutton(c, text=txt, variable=self.formula_var, value=i,
                           bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W)

        self.antimatter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="m+ и m- взаимодействуют", variable=self.antimatter_var,
                       bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W, pady=(5, 0))

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Label(c, text="Параметры силы", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        a_frame = tk.Frame(c, bg="#1e1e1e")
        a_frame.pack(fill=tk.X, pady=2)
        tk.Label(a_frame, text="a", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.a_slider = tk.Scale(a_frame, from_=-9, to=9, resolution=1,
                                 orient=tk.HORIZONTAL, showvalue=0,
                                 bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.a_slider.set(1)
        self.a_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.a_label = tk.Label(a_frame, text="1", fg="white", bg="#1e1e1e", width=6)
        self.a_label.pack(side=tk.LEFT, padx=5)
        self.a_slider.config(command=lambda v: self.a_label.config(text=str(int(float(v)))))

        b_frame = tk.Frame(c, bg="#1e1e1e")
        b_frame.pack(fill=tk.X, pady=2)
        tk.Label(b_frame, text="b", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.b_slider = tk.Scale(b_frame, from_=-50, to=50, resolution=1,
                                 orient=tk.HORIZONTAL, showvalue=0,
                                 bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.b_slider.set(0)
        self.b_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.b_label = tk.Label(b_frame, text="0", fg="white", bg="#1e1e1e", width=6)
        self.b_label.pack(side=tk.LEFT, padx=5)
        self.b_slider.config(command=lambda v: self.b_label.config(text=str(int(float(v)))))

        c_frame = tk.Frame(c, bg="#1e1e1e")
        c_frame.pack(fill=tk.X, pady=2)
        tk.Label(c_frame, text="c", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.c_slider = tk.Scale(c_frame, from_=1, to=100, resolution=1,
                                 orient=tk.HORIZONTAL, showvalue=0,
                                 bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.c_slider.set(1)
        self.c_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.c_label = tk.Label(c_frame, text="1", fg="white", bg="#1e1e1e", width=6)
        self.c_label.pack(side=tk.LEFT, padx=5)
        self.c_slider.config(command=lambda v: self.c_label.config(text=str(int(float(v)))))

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Button(c, text="Применить настройки", command=self.apply_settings,
                  width=20, bg="#3e3e3e", fg="white").pack(pady=5)

        tk.Frame(c, height=1, bg="#1e1e1e").pack(fill=tk.Y, expand=True)

    def create_random_tab(self):
        tab = tk.Frame(self.panel, bg="#1e1e1e")
        self.panel.add(tab, text="Случайность")

        c = tk.Frame(tab, bg="#1e1e1e")
        c.pack(fill=tk.Y, expand=True, padx=10, pady=10)

        tk.Label(c, text="Количество пар на 100 шагов",
                 fg="white", bg="#1e1e1e", font=("Arial", 10)).pack(pady=(10, 0))

        rnd_frame = tk.Frame(c, bg="#1e1e1e")
        rnd_frame.pack(pady=5, fill=tk.X)

        self.rnd_slider = tk.Scale(rnd_frame, from_=0, to=1000, resolution=1,
                                   orient=tk.HORIZONTAL, showvalue=0,
                                   bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.rnd_slider.set(0)
        self.rnd_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.rnd_label = tk.Label(rnd_frame, text="0", fg="white", bg="#1e1e1e", width=5)
        self.rnd_label.pack(side=tk.LEFT)
        self.rnd_slider.config(command=lambda v: self.rnd_label.config(text=str(int(float(v)))))

        tk.Button(c, text="Применить настройки", command=self.apply_random_settings,
                  width=20, bg="#3e3e3e", fg="white").pack(pady=15)

        tk.Frame(c, height=1, bg="#1e1e1e").pack(fill=tk.Y, expand=True)

    def toggle_start_stop(self):
        if self.sim.running:
            self.sim.running = False
            self.start_stop_btn.config(text="Старт")
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
        else:
            self.sim.running = True
            self.start_stop_btn.config(text="Стоп")
            self.animation_loop()

    def reset_sim(self):
        self.sim.running = False
        self.start_stop_btn.config(text="Старт")
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.sim.reset()
        self.update_stats()
        self.renderer.draw(self.sim.particles)

    def add_particles(self):
        count = int(self.add_slider.get())
        self.sim.add_particles(count)
        self.update_stats()
        self.renderer.draw(self.sim.particles)

    def set_speed(self, val):
        self.steps_per_frame = int(float(val))
        self.speed_label.config(text=str(self.steps_per_frame))

    def animation_loop(self):
        if not self.sim.running:
            return
        for _ in range(self.steps_per_frame):
            self.sim.update_step()
        self.update_stats()
        self.renderer.draw(self.sim.particles)
        self.after_id = self.root.after(33, self.animation_loop)

    def update_stats(self):
        stats = self.sim.get_stats()
        self.count_label.config(text=f"Точек: {stats['count']}")
        self.step_label.config(text=f"Шаг: {stats['steps']}")

        top = self.sim.get_top_masses(15)
        for i in range(15):
            if i < len(top):
                m = top[i]
                sign = "+" if m >= 0 else ""
                self.top_labels[i].config(text=f"{i+1}. {sign}{m}")
            else:
                self.top_labels[i].config(text=f"{i+1}. ---")

    def apply_settings(self):
        self.world.width = float(self.l_slider.get())
        self.world.height = float(self.h_slider.get())
        self.world.boundary = self.boundary_var.get()
        self.merger.enabled = self.merge_var.get()

        self.physics.formula_idx = self.formula_var.get()
        self.physics.a = float(self.a_slider.get())
        self.physics.b = float(self.b_slider.get())
        self.physics.c = float(self.c_slider.get())
        self.physics.antimatter_interaction = self.antimatter_var.get()
        self.merger.antimatter_interaction = self.antimatter_var.get()

        self.speed_slider.set(1)
        self.steps_per_frame = 1
        self.speed_label.config(text="1")

        self.reset_sim()

    def apply_random_settings(self):
        val = int(self.rnd_slider.get())
        self.sim.random_gen.set_params(val)


if __name__ == "__main__":
    root = tk.Tk()
    app = UniverseApp(root)
    root.mainloop()


# In[ ]:




