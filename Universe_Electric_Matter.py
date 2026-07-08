#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import sys
import random


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'mass', 'charge')

    def __init__(self, x, y, vx=0.0, vy=0.0, mass=1.0, charge=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.charge = charge


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
    def __init__(self,
                 mass_formula_idx=0, mass_a=1.0, mass_b=0.0, mass_c=1.0,
                 charge_formula_idx=0, charge_a=-1.0, charge_b=0.0, charge_c=1.0,
                 softening=0.5):
        self.mass_formula_idx = mass_formula_idx
        self.mass_a = mass_a
        self.mass_b = mass_b
        self.mass_c = mass_c

        self.charge_formula_idx = charge_formula_idx
        self.charge_a = charge_a
        self.charge_b = charge_b
        self.charge_c = charge_c

        self.softening = softening

    def compute_force_mass(self, p1, p2, dist, dist2):
        m1, m2 = p1.mass, p2.mass
        coeff = self.mass_a * (10.0 ** self.mass_b)
        mass_product = m1 * m2

        if mass_product == 0:
            return 0.0

        if self.mass_formula_idx == 0:
            f = coeff * mass_product / dist
        elif self.mass_formula_idx == 1:
            f = coeff * mass_product / dist2
        elif self.mass_formula_idx == 2:
            f = -coeff * dist
        elif self.mass_formula_idx == 3:
            f = -coeff * mass_product * dist
        elif self.mass_formula_idx == 4:
            f = coeff * mass_product * math.exp(-dist / self.mass_c) / dist
        elif self.mass_formula_idx == 5:
            f = coeff * mass_product * math.exp(-dist / self.mass_c) / dist2
        else:
            f = 0.0

        return f

    def compute_force_charge(self, p1, p2, dist, dist2):
        q1, q2 = p1.charge, p2.charge
        coeff = self.charge_a * (10.0 ** self.charge_b)
        charge_product = q1 * q2

        if charge_product == 0:
            return 0.0

        if self.charge_formula_idx == 0:
            f = coeff * charge_product / dist
        elif self.charge_formula_idx == 1:
            f = coeff * charge_product / dist2
        elif self.charge_formula_idx == 2:
            f = -coeff * dist
        elif self.charge_formula_idx == 3:
            f = -coeff * charge_product * dist
        elif self.charge_formula_idx == 4:
            f = coeff * charge_product * math.exp(-dist / self.charge_c) / dist
        elif self.charge_formula_idx == 5:
            f = coeff * charge_product * math.exp(-dist / self.charge_c) / dist2
        else:
            f = 0.0

        return f

    def compute_force(self, p1, p2, world):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dx, dy = world.get_minimum_image(dx, dy)

        dist2 = dx * dx + dy * dy + self.softening * self.softening
        dist = math.sqrt(dist2)
        if dist < 0.0001:
            return 0.0, 0.0

        f_mass = self.compute_force_mass(p1, p2, dist, dist2)
        f_charge = self.compute_force_charge(p1, p2, dist, dist2)
        f = f_mass + f_charge

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
            if particles[i].mass != 0.0:
                ax[i] = fx_total / particles[i].mass
                ay[i] = fy_total / particles[i].mass
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
    def __init__(self, enabled=True, density=1.0):
        self.enabled = enabled
        self.density = density

    def get_radius(self, mass):
        if mass <= 0:
            return 0.1
        return math.sqrt(mass / (math.pi * self.density))

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

            total_mass = 0.0
            total_charge = 0.0
            cx = cy = px = py = 0.0
            for idx in indices:
                p = particles[idx]
                total_mass += p.mass
                total_charge += p.charge
                cx += p.x * p.mass
                cy += p.y * p.mass
                px += p.vx * p.mass
                py += p.vy * p.mass

            if total_mass != 0.0:
                new_particles.append(Particle(
                    cx / total_mass, cy / total_mass,
                    px / total_mass, py / total_mass,
                    total_mass, total_charge
                ))

        return new_particles


class RandomGenerator:
    def __init__(self):
        self.neutral_particles_per_100 = 0
        self.charged_particles_per_100 = 0
        self.charged_pairs = True
        self.neutral_schedule = []
        self.charged_schedule = []
        self.current_step_in_window = 0

    def set_params(self, neutral_particles, charged_particles, charged_pairs=True):
        self.neutral_particles_per_100 = neutral_particles
        self.charged_particles_per_100 = charged_particles
        self.charged_pairs = charged_pairs
        self.reset_schedules()

    def reset_schedules(self):
        self.neutral_schedule = []
        self.charged_schedule = []
        self.current_step_in_window = 0

        if self.neutral_particles_per_100 > 0:
            self.neutral_schedule = [random.randint(1, 100) for _ in range(self.neutral_particles_per_100)]
        if self.charged_particles_per_100 > 0:
            self.charged_schedule = [random.randint(1, 100) for _ in range(self.charged_particles_per_100)]

    def get_counts_for_step(self):
        if self.neutral_particles_per_100 == 0 and self.charged_particles_per_100 == 0:
            return 0, 0

        self.current_step_in_window += 1

        if self.current_step_in_window > 100:
            self.current_step_in_window = 1
            if self.neutral_particles_per_100 > 0:
                self.neutral_schedule = [random.randint(1, 100) for _ in range(self.neutral_particles_per_100)]
            if self.charged_particles_per_100 > 0:
                self.charged_schedule = [random.randint(1, 100) for _ in range(self.charged_particles_per_100)]

        neutral_count = self.neutral_schedule.count(self.current_step_in_window)
        charged_count = self.charged_schedule.count(self.current_step_in_window)

        return neutral_count, charged_count


class Renderer:
    def __init__(self, canvas, world, density=1.0):
        self.canvas = canvas
        self.world = world
        self.density = density
        self.min_radius = 1.0

    def get_radius(self, mass):
        if mass <= 0:
            return self.min_radius
        return math.sqrt(mass / (math.pi * self.density))

    def mass_to_color(self, mass):
        if mass <= 1.0:
            return "#FFE4B5"
        if mass >= 100.0:
            return "#8B4513"

        t = (mass - 1.0) / 99.0
        r1, g1, b1 = 255, 228, 181
        r2, g2, b2 = 139, 69, 19
        return f"#{int(r1 + (r2 - r1) * t):02x}{int(g1 + (g2 - g1) * t):02x}{int(b1 + (b2 - b1) * t):02x}"

    def charge_outline_color(self, charge):
        if charge > 0:
            return "#FF0000"
        elif charge < 0:
            return "#0000FF"
        else:
            return "#808080"

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

            fill_color = self.mass_to_color(p.mass)
            outline_color = self.charge_outline_color(p.charge)

            self.canvas.create_oval(
                screen_x - radius, screen_y - radius,
                screen_x + radius, screen_y + radius,
                fill=fill_color, outline=outline_color, width=1
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
        self.random_gen.reset_schedules()

    def add_neutral_particles(self, count):
        for _ in range(count):
            x = random.uniform(0.0, self.world.width)
            y = random.uniform(0.0, self.world.height)
            self.particles.append(Particle(x, y, mass=1.0, charge=0.0))

    def add_charged_particles(self, count, pairs=True):
        if count <= 0:
            return

        if pairs:
            half = count // 2
            for _ in range(half):
                x = random.uniform(0.0, self.world.width)
                y = random.uniform(0.0, self.world.height)
                angle = random.uniform(0.0, 2.0 * math.pi)
                dist = random.uniform(1.0, 3.0)

                x1 = max(0.0, min(self.world.width, x + dist * math.cos(angle)))
                y1 = max(0.0, min(self.world.height, y + dist * math.sin(angle)))
                x2 = max(0.0, min(self.world.width, x - dist * math.cos(angle)))
                y2 = max(0.0, min(self.world.height, y - dist * math.sin(angle)))

                self.particles.append(Particle(x1, y1, mass=1.0, charge=1.0))
                self.particles.append(Particle(x2, y2, mass=1.0, charge=-1.0))
        else:
            half = count // 2
            for i in range(count):
                x = random.uniform(0.0, self.world.width)
                y = random.uniform(0.0, self.world.height)
                charge = 1.0 if i < half else -1.0
                self.particles.append(Particle(x, y, mass=1.0, charge=charge))

    def update_step(self):
        if not self.running:
            return

        self.physics.euler_step(self.particles, self.world, self.dt)

        if self.world.boundary == 'open':
            self.particles = [p for p in self.particles if self.world.is_inside(p.x, p.y)]
        else:
            self.world.apply_boundary(self.particles)

        self.particles = self.merger.merge(self.particles, self.world)

        neutral_count, charged_count = self.random_gen.get_counts_for_step()
        if neutral_count > 0:
            self.add_neutral_particles(neutral_count)
        if charged_count > 0:
            self.add_charged_particles(charged_count, self.random_gen.charged_pairs)

        self.step_counter += 1

    def get_stats(self):
        return {'count': len(self.particles), 'steps': self.step_counter}

    def get_top_sums(self, n=10):
        sorted_p = sorted(self.particles, key=lambda p: abs(p.mass) + abs(p.charge), reverse=True)
        return [(p.mass, p.charge) for p in sorted_p[:n]]


class UniverseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Insane Universe. Electric Matter")

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
        self.merger = Merger()
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

        tk.Label(c, text="ДОБАВЛЕНИЕ ЧАСТИЦ", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        tk.Label(c, text="Нейтральные частицы (q = 0)", fg="white", bg="#1e1e1e",
                 font=("Arial", 9)).pack(pady=(5, 0), anchor="w")
        add_frame = tk.Frame(c, bg="#1e1e1e")
        add_frame.pack(pady=3, fill=tk.X)
        tk.Label(add_frame, text="Частиц", fg="white", bg="#1e1e1e", width=6).pack(side=tk.LEFT)
        self.neutral_slider = tk.Scale(add_frame, from_=1, to=1000, resolution=1,
                                       orient=tk.HORIZONTAL, showvalue=0,
                                       bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.neutral_slider.set(100)
        self.neutral_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.neutral_label = tk.Label(add_frame, text="100", fg="white", bg="#1e1e1e", width=5)
        self.neutral_label.pack(side=tk.LEFT)
        self.neutral_slider.config(command=lambda v: self.neutral_label.config(text=str(int(float(v)))))
        tk.Button(c, text="Добавить", command=self.add_neutral_particles,
                  width=18, bg="#2a5a2a", fg="white").pack(pady=(0, 5))

        tk.Label(c, text="Заряженные частицы (q = ±1)", fg="white", bg="#1e1e1e",
                 font=("Arial", 9)).pack(pady=(5, 0), anchor="w")
        add_frame = tk.Frame(c, bg="#1e1e1e")
        add_frame.pack(pady=3, fill=tk.X)
        tk.Label(add_frame, text="Частиц", fg="white", bg="#1e1e1e", width=6).pack(side=tk.LEFT)
        self.charged_slider = tk.Scale(add_frame, from_=0, to=1000, resolution=2,
                                       orient=tk.HORIZONTAL, showvalue=0,
                                       bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.charged_slider.set(100)
        self.charged_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.charged_label = tk.Label(add_frame, text="100", fg="white", bg="#1e1e1e", width=5)
        self.charged_label.pack(side=tk.LEFT)
        self.charged_slider.config(command=lambda v: self.charged_label.config(text=str(int(float(v)))))

        self.charged_pairs_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Располагать парами", variable=self.charged_pairs_var,
                       bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W, pady=(2, 0))

        tk.Button(c, text="Добавить", command=self.add_charged_particles,
                  width=18, bg="#2a5a2a", fg="white").pack(pady=(0, 5))

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        self.count_label = tk.Label(c, text="Точек: 0", fg="#88ff88", bg="#1e1e1e",
                                    font=("Arial", 10, "bold"), anchor="w")
        self.count_label.pack(fill=tk.X, pady=1)

        self.step_label = tk.Label(c, text="Шаг: 0", fg="#ffff88", bg="#1e1e1e",
                                   font=("Arial", 10, "bold"), anchor="w")
        self.step_label.pack(fill=tk.X, pady=1)

        tk.Label(c, text="Топ-10 |M| + |Q|", fg="white", bg="#1e1e1e",
                 font=("Arial", 10)).pack(pady=(10, 0))

        self.top_labels = []
        for i in range(10):
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

        tk.Label(c, text="Формулы силы", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        self.formula_notebook = ttk.Notebook(c)
        self.formula_notebook.pack(fill=tk.X, pady=5)

        self.create_mass_tab_settings()
        self.create_charge_tab_settings()

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Button(c, text="Применить настройки", command=self.apply_settings,
                  width=20, bg="#3e3e3e", fg="white").pack(pady=5)

        tk.Frame(c, height=1, bg="#1e1e1e").pack(fill=tk.Y, expand=True)

    def create_mass_tab_settings(self):
        tab = tk.Frame(self.formula_notebook, bg="#1e1e1e")
        self.formula_notebook.add(tab, text="Масса")

        c = tk.Frame(tab, bg="#1e1e1e")
        c.pack(fill=tk.X, padx=5, pady=5)

        self.mass_formula_var = tk.IntVar(value=0)
        formulas = [
            "a*10^b * m1*m2 / r",
            "a*10^b * m1*m2 / r^2",
            "-a*10^b * r",
            "-a*10^b * m1*m2 * r",
            "a*10^b * m1*m2 * e^(-r/c) / r",
            "a*10^b * m1*m2 * e^(-r/c) / r^2",
        ]
        for i, txt in enumerate(formulas):
            tk.Radiobutton(c, text=txt, variable=self.mass_formula_var, value=i,
                           bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W)

        a_frame = tk.Frame(c, bg="#1e1e1e")
        a_frame.pack(fill=tk.X, pady=2)
        tk.Label(a_frame, text="a", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.mass_a_slider = tk.Scale(a_frame, from_=-9, to=9, resolution=0.1,
                                      orient=tk.HORIZONTAL, showvalue=0,
                                      bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.mass_a_slider.set(1.0)
        self.mass_a_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mass_a_label = tk.Label(a_frame, text="1.0", fg="white", bg="#1e1e1e", width=6)
        self.mass_a_label.pack(side=tk.LEFT, padx=5)
        self.mass_a_slider.config(command=lambda v: self.mass_a_label.config(text=f"{float(v):.1f}"))

        b_frame = tk.Frame(c, bg="#1e1e1e")
        b_frame.pack(fill=tk.X, pady=2)
        tk.Label(b_frame, text="b", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.mass_b_slider = tk.Scale(b_frame, from_=-50, to=50, resolution=1,
                                      orient=tk.HORIZONTAL, showvalue=0,
                                      bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.mass_b_slider.set(0)
        self.mass_b_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mass_b_label = tk.Label(b_frame, text="0", fg="white", bg="#1e1e1e", width=6)
        self.mass_b_label.pack(side=tk.LEFT, padx=5)
        self.mass_b_slider.config(command=lambda v: self.mass_b_label.config(text=str(int(float(v)))))

        c_frame = tk.Frame(c, bg="#1e1e1e")
        c_frame.pack(fill=tk.X, pady=2)
        tk.Label(c_frame, text="c", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.mass_c_slider = tk.Scale(c_frame, from_=1, to=100, resolution=1,
                                      orient=tk.HORIZONTAL, showvalue=0,
                                      bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.mass_c_slider.set(1)
        self.mass_c_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mass_c_label = tk.Label(c_frame, text="1", fg="white", bg="#1e1e1e", width=6)
        self.mass_c_label.pack(side=tk.LEFT, padx=5)
        self.mass_c_slider.config(command=lambda v: self.mass_c_label.config(text=str(int(float(v)))))

    def create_charge_tab_settings(self):
        tab = tk.Frame(self.formula_notebook, bg="#1e1e1e")
        self.formula_notebook.add(tab, text="Заряд")

        c = tk.Frame(tab, bg="#1e1e1e")
        c.pack(fill=tk.X, padx=5, pady=5)

        self.charge_formula_var = tk.IntVar(value=0)
        formulas = [
            "a*10^b * q1*q2 / r",
            "a*10^b * q1*q2 / r^2",
            "-a*10^b * r",
            "-a*10^b * q1*q2 * r",
            "a*10^b * q1*q2 * e^(-r/c) / r",
            "a*10^b * q1*q2 * e^(-r/c) / r^2",
        ]
        for i, txt in enumerate(formulas):
            tk.Radiobutton(c, text=txt, variable=self.charge_formula_var, value=i,
                           bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W)

        a_frame = tk.Frame(c, bg="#1e1e1e")
        a_frame.pack(fill=tk.X, pady=2)
        tk.Label(a_frame, text="a", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.charge_a_slider = tk.Scale(a_frame, from_=-9, to=9, resolution=0.1,
                                        orient=tk.HORIZONTAL, showvalue=0,
                                        bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.charge_a_slider.set(-1.0)
        self.charge_a_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.charge_a_label = tk.Label(a_frame, text="-1.0", fg="white", bg="#1e1e1e", width=6)
        self.charge_a_label.pack(side=tk.LEFT, padx=5)
        self.charge_a_slider.config(command=lambda v: self.charge_a_label.config(text=f"{float(v):.1f}"))

        b_frame = tk.Frame(c, bg="#1e1e1e")
        b_frame.pack(fill=tk.X, pady=2)
        tk.Label(b_frame, text="b", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.charge_b_slider = tk.Scale(b_frame, from_=-50, to=50, resolution=1,
                                        orient=tk.HORIZONTAL, showvalue=0,
                                        bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.charge_b_slider.set(0)
        self.charge_b_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.charge_b_label = tk.Label(b_frame, text="0", fg="white", bg="#1e1e1e", width=6)
        self.charge_b_label.pack(side=tk.LEFT, padx=5)
        self.charge_b_slider.config(command=lambda v: self.charge_b_label.config(text=str(int(float(v)))))

        c_frame = tk.Frame(c, bg="#1e1e1e")
        c_frame.pack(fill=tk.X, pady=2)
        tk.Label(c_frame, text="c", fg="white", bg="#1e1e1e", width=3).pack(side=tk.LEFT)
        self.charge_c_slider = tk.Scale(c_frame, from_=1, to=100, resolution=1,
                                        orient=tk.HORIZONTAL, showvalue=0,
                                        bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.charge_c_slider.set(1)
        self.charge_c_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.charge_c_label = tk.Label(c_frame, text="1", fg="white", bg="#1e1e1e", width=6)
        self.charge_c_label.pack(side=tk.LEFT, padx=5)
        self.charge_c_slider.config(command=lambda v: self.charge_c_label.config(text=str(int(float(v)))))

    def create_random_tab(self):
        tab = tk.Frame(self.panel, bg="#1e1e1e")
        self.panel.add(tab, text="Случайность")

        c = tk.Frame(tab, bg="#1e1e1e")
        c.pack(fill=tk.Y, expand=True, padx=10, pady=10)

        tk.Label(c, text="Нейтральные частицы (q = 0)", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        tk.Label(c, text="Частиц на 100 шагов", fg="white", bg="#1e1e1e",
                 font=("Arial", 9)).pack(pady=(2, 0), anchor="w")

        frame_neutral = tk.Frame(c, bg="#1e1e1e")
        frame_neutral.pack(pady=3, fill=tk.X)
        self.rnd_neutral_slider = tk.Scale(frame_neutral, from_=0, to=100, resolution=1,
                                           orient=tk.HORIZONTAL, showvalue=0,
                                           bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.rnd_neutral_slider.set(0)
        self.rnd_neutral_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rnd_neutral_label = tk.Label(frame_neutral, text="0", fg="white", bg="#1e1e1e", width=5)
        self.rnd_neutral_label.pack(side=tk.LEFT)
        self.rnd_neutral_slider.config(command=lambda v: self.rnd_neutral_label.config(text=str(int(float(v)))))

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=5, fill=tk.X)

        tk.Label(c, text="Заряженные частицы (q = ±1)", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(pady=(5, 0), anchor="w")

        tk.Label(c, text="Частиц на 100 шагов", fg="white", bg="#1e1e1e",
                 font=("Arial", 9)).pack(pady=(2, 0), anchor="w")

        frame_charged = tk.Frame(c, bg="#1e1e1e")
        frame_charged.pack(pady=3, fill=tk.X)
        self.rnd_charged_slider = tk.Scale(frame_charged, from_=0, to=100, resolution=2,
                                           orient=tk.HORIZONTAL, showvalue=0,
                                           bg="#1e1e1e", fg="white", highlightbackground="#1e1e1e")
        self.rnd_charged_slider.set(0)
        self.rnd_charged_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rnd_charged_label = tk.Label(frame_charged, text="0", fg="white", bg="#1e1e1e", width=5)
        self.rnd_charged_label.pack(side=tk.LEFT)
        self.rnd_charged_slider.config(command=lambda v: self.rnd_charged_label.config(text=str(int(float(v)))))

        self.rnd_charged_pairs_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Располагать парами", variable=self.rnd_charged_pairs_var,
                       bg="#1e1e1e", fg="white", selectcolor="#1e1e1e").pack(anchor=tk.W, pady=(5, 0))

        tk.Frame(c, height=2, bg="#3e3e3e").pack(pady=10, fill=tk.X)

        tk.Button(c, text="Применить настройки", command=self.apply_settings,
                  width=20, bg="#3e3e3e", fg="white").pack(pady=5)

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

    def add_neutral_particles(self):
        count = int(self.neutral_slider.get())
        self.sim.add_neutral_particles(count)
        self.update_stats()
        self.renderer.draw(self.sim.particles)

    def add_charged_particles(self):
        count = int(self.charged_slider.get())
        pairs = self.charged_pairs_var.get()
        self.sim.add_charged_particles(count, pairs)
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

        top = self.sim.get_top_sums(10)
        for i in range(10):
            if i < len(top):
                m, q = top[i]
                sign_m = "+" if m >= 0 else ""
                sign_q = "+" if q >= 0 else ""
                self.top_labels[i].config(
                    text=f"{i+1}. m={sign_m}{int(m)}, q={sign_q}{int(q)} (сумма {int(abs(m)+abs(q))})"
                )
            else:
                self.top_labels[i].config(text=f"{i+1}. ---")

    def apply_settings(self):
        self.world.width = float(self.l_slider.get())
        self.world.height = float(self.h_slider.get())
        self.world.boundary = self.boundary_var.get()
        self.merger.enabled = self.merge_var.get()

        self.physics.mass_formula_idx = self.mass_formula_var.get()
        self.physics.mass_a = float(self.mass_a_slider.get())
        self.physics.mass_b = float(self.mass_b_slider.get())
        self.physics.mass_c = float(self.mass_c_slider.get())

        self.physics.charge_formula_idx = self.charge_formula_var.get()
        self.physics.charge_a = float(self.charge_a_slider.get())
        self.physics.charge_b = float(self.charge_b_slider.get())
        self.physics.charge_c = float(self.charge_c_slider.get())

        neutral = int(self.rnd_neutral_slider.get())
        charged = int(self.rnd_charged_slider.get())
        charged_pairs = self.rnd_charged_pairs_var.get()
        self.sim.random_gen.set_params(neutral, charged, charged_pairs)

        self.speed_slider.set(1)
        self.steps_per_frame = 1
        self.speed_label.config(text="1")

        self.reset_sim()


if __name__ == "__main__":
    root = tk.Tk()
    app = UniverseApp(root)
    root.mainloop()


# In[ ]:




