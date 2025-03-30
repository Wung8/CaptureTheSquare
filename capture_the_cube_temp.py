import pygame
import numpy as np
import math, random, time

# Initialize pygame
pygame.init()

background = pygame.image.load("background.png")
arrow_img = pygame.image.load("arrow.png")

# Game settings
scale = 5
tile_size = 9
respawn_time = 50

# Colors
colors = {
    "light_grey": (180, 180, 180),
    "grey": (130, 130, 130),
    "diamond": (50, 200, 255),
    "club": (50, 230, 50),
    "spade": (150, 50, 230),
    "heart": (230, 70, 70),
    "health_bg": (100, 100, 100),
    "health_fg": (0, 255, 0),
    
    "p0_bg": (142, 152, 179), # light blue
    "p0_fg": (89, 113, 179), # blue

    "p1_bg": (128, 166, 123), # light green
    "p1_fg": (78, 142, 72), # green
    
    "p2_bg": (146, 136, 169), # light purple
    "p2_fg": (113, 88, 149), # purple
    
    "p3_bg": (182, 142, 142), # light red
    "p3_fg": (162, 70, 70), # red
}

# Map layout
map_layout = '''
...X.......X...
...X...X...X...
.......XXX.....
XX.....X.....XX
....XX.X..X....
..X.......X....
..X............
.XXXX.....XXXX.
............X..
....X.......X..
....X..X.XX....
XX.....X.....XX
.....XXX.......
...X...X...X...
...X.......X...
'''.strip().split('\n')

# Generate wall hitbox
wall_hitbox = np.zeros((15, 15))
for y, row in enumerate(map_layout):
    for x, tile in enumerate(row):
        if tile == 'X':
            wall_hitbox[y][x] = 1
wall_hitbox = np.repeat(np.repeat(wall_hitbox, tile_size, axis=0), tile_size, axis=1)
temp_wall_hitbox = wall_hitbox.copy()
for y, row in enumerate(temp_wall_hitbox):
    for x, tile in enumerate(row):
        if tile == 1:
            for dy in range(-1,2):
                for dx in range(-1,2):
                    try:
                        wall_hitbox[y+dy][x+dx] = 1
                    except:
                        pass

base_hitbox = np.zeros((135,135),dtype=np.uint8)
for y, row in enumerate(temp_wall_hitbox):
    for x, tile in enumerate(row):
        if y < 36 and x < 36:
            base_hitbox[y][x] = 1
        if y < 36 and x > 99:
            base_hitbox[y][x] = 2
        if y > 99 and x < 36:
            base_hitbox[y][x] = 3
        if y > 99 and x > 99:
            base_hitbox[y][x] = 4

# Weapon configurations: [max_health, damage, attack_speed, hitbox_size]
# hitbox is top left to bottom right, facing the right
# weapon_configs = {
#     "sword": (20, 4, 1.5, ((0, -10), (10, 10))),  # Larger hitbox
#     "bow": (10, 3, 1.0, ((0, 0), (30, 2))),  # Small hitbox (point)
#     "shield": (20, 2, 0.6, ((0, -10), (5, 10))),  # Medium-sized hitbox
# }

team_idxs = {
    "diamond":0,
    "club":1,
    "spade":2,
    "heart":3
}

weapon_configs = {
    "sword": (20, 4, 6, 11),  # Larger hitbox
    "bow": (10, 3, 12, -1),  # Small hitbox (point)
    "shield": (20, 2, 16, 8.5),  # Medium-sized hitbox
}

# Player spawn locations
spawn_configs = {
    "diamond": (1, 1),
    "club": (13, 1),
    "spade": (1, 13),
    "heart": (13, 13),
    #"diamond": (6, 6),
    #"club": (9, 6),
    #"spade": (6, 9),
    #"heart": (9, 9),

}

arrow_speed = 2
class Arrow:
    def __init__(self, parent, player, team, pos, angle):
        angle_rad = math.radians(angle)
        self.player = player
        self.angle = angle
        self.parent = parent
        self.team = team
        self.pos = pos
        self.velocity = np.multiply((math.cos(angle_rad), math.sin(angle_rad)),2)

    def step(self):
        self.pos = np.add(self.pos, self.velocity)
        x,y = self.pos
        x,y = round(x), round(y)
        if x < 0 or x > 134 or y < 0 or y > 134 or wall_hitbox[y][x]:
            self.team = -1
            return
        for target in self.parent.players:
            if target.respawn_timer != 0: continue
            tx, ty = target.pos
            if target.weapon == "shield":
                rad = 6
            else:
                rad = 2.5
            if math.dist(self.pos, (tx,ty)) < rad and target.team != self.team:
                if target.weapon != "shield":
                    target.health = max(0, target.health - 3)
                else:
                    target_vec = self.velocity
                    shield_angle_rad = math.radians(target.angle)
                    shield_normal_vec = (-math.cos(shield_angle_rad), -math.sin(shield_angle_rad))
                    if np.dot(shield_normal_vec, target_vec) >= 0:
                        target.health = max(0, target.health - 3/6)
                        self.player.r -= 0.6
                    else:
                        target.health = max(0, target.health - 3)
                self.player.r += 1
                self.team = -1
        

    def render(self):
        x,y = self.pos
        x,y = round(x), round(y)
        rotated_weapon = pygame.transform.rotate(arrow_img, -self.angle)
        weapon_rect = rotated_weapon.get_rect(center=(x, y))
        self.parent.surface.blit(rotated_weapon, weapon_rect.topleft)
        
def norm(v):
    return v / np.sqrt(np.sum(v**2))

class Player:
    def __init__(self, parent, team, weapon, _id):
        self.id = _id
        self.parent = parent
        self.team = team
        self.weapon = weapon
        self.reset()

        # Load weapon images
        self.passive_image = pygame.image.load(f"{weapon}_passive.png")
        self.active_image = pygame.image.load(f"{weapon}_active.png")
        self.current_image = self.passive_image
        self.angle = 0

    def reset(self, player_idx=-1):
        """Initialize player stats"""
        self.player_idx = player_idx
        self.max_health, self.damage, self.max_attack_cooldown, self.hitbox_size = weapon_configs[self.weapon]
        self.health = self.max_health
        self.pos = np.multiply(spawn_configs[self.team], tile_size) + 4
        self.attack_cooldown = 0
        self.respawn_timer = respawn_time

    def step(self, action):
        self.r = 0
        if self.respawn_timer != 0:
            self.respawn_timer -= 1
            #self.r = -0.05
            return
        
        """Move the player"""
        x, y = self.pos
        dx, dy = np.subtract(action[:2], 1)
        if (x + dx > 0 and x + dx < 134) and wall_hitbox[y][x + dx] != 1:
            x += dx
        if (y + dy > 0 and y + dy < 134) and wall_hitbox[y + dy][x] != 1:
            y += dy
            
        v = np.subtract(np.array((67,67),dtype=np.float32), self.pos)
        nz = v != 0
        v[nz] = norm(v[nz])
        self.r += np.dot(np.subtract((x,y),self.pos), v) / 67 * 2.5
        
        self.pos = x, y

        if base_hitbox[y][x] != 0 and base_hitbox[y][x] != team_idxs[self.team]+1:
            self.health -= 1

        self.angle = action[3]*90 + action[4]*18-36
        if self.weapon == "bow" and self.id != self.parent.player_idx:
            targets = [target for target in self.parent.players if target.team != self.team]
            target = targets[np.argmin([math.dist(self.pos,target.pos) for target in targets])]
            dx, dy = np.subtract(target.pos, self.pos)
            self.angle = math.degrees(math.atan2(dy, dx))

        if action[2] == 1 and self.attack_cooldown == 0:
            self.r -= 0.01
            self.current_image = self.active_image
            self.attack_cooldown = self.max_attack_cooldown

            if self.weapon == "bow":
                self.parent.arrows.append(Arrow(self.parent,self,self.team,self.pos,self.angle))
            else:
                for target in self.parent.players:
                    if target.respawn_timer != 0: continue
                    if target.team != self.team:
                        tx, ty = target.pos
                        angle_rad = math.radians(self.angle)
                        normal_vec = (math.cos(angle_rad), math.sin(angle_rad))
                        target_vec = (tx-x, ty-y)
                        if math.dist((x, y), (tx, ty)) <= self.hitbox_size and np.dot(normal_vec, target_vec) >= 0:
                            if target.weapon != "shield":
                                target.health = max(0, target.health - self.damage)
                            else:
                                shield_angle_rad = math.radians(target.angle)
                                shield_normal_vec = (-math.cos(shield_angle_rad), -math.sin(shield_angle_rad))
                                if np.dot(shield_normal_vec, target_vec) >= 0:
                                    target.health = max(0, target.health - self.damage/6)
                                    self.r -= 0.6
                                else:
                                    target.health = max(0, target.health - self.damage)
                            self.r += 1
        else:
            self.current_image = self.passive_image
            self.attack_cooldown = max(self.attack_cooldown-1,0)
 

    def render(self):
        if self.respawn_timer != 0:
            return
        
        x, y = self.pos

        # self
        pygame.draw.rect(
            self.parent.surface,
            colors[self.team],
            (x-1, y-1, 3, 3)
        )   

        rotated_weapon = pygame.transform.rotate(self.current_image, -self.angle)
        weapon_rect = rotated_weapon.get_rect(center=(x, y))
        self.parent.surface.blit(rotated_weapon, weapon_rect.topleft)

        # Draw health bar
        bar_width = 10
        health_ratio = self.health / self.max_health    
        pygame.draw.rect(self.parent.surface, colors["health_bg"], (x - 5, y - 8, bar_width, 1))
        pygame.draw.rect(self.parent.surface, colors["health_fg"], (x - 5, y - 8, int(bar_width * health_ratio), 1))

    def get_obs(self):
        sr = 24
        x,y = self.pos
        x_padded, y_padded = x + 32, y + 32

        obs = self.parent.padded_surface_array[:,y_padded-sr:y_padded+sr,x_padded-sr:x_padded+sr]
        return obs

class CTCEnvironment:
    def __init__(self, render_mode="None"):
        self.map_size = 135, 135
        self.render_mode = render_mode
        
        if render_mode == "None":
            self.screen_size = -1, -1
        elif render_mode == "human":
            self.screen_size = 40, 40
        elif render_mode == "god":
            self.screen_size = self.map_size
        else:
            raise ValueError("Invalid render mode")

        self.surface = pygame.Surface(self.map_size, pygame.SRCALPHA)
        if render_mode != "None":
            self.screen = pygame.display.set_mode(np.multiply(self.screen_size, scale))
            self.clock = pygame.time.Clock()

        # Create players
        self.players = [
            Player(self, "diamond", "bow", 0),
            Player(self, "diamond", "sword", 1),
            Player(self, "club", "bow", 2),
            Player(self, "club", "bow", 3),
            Player(self, "spade", "sword", 4),
            Player(self, "spade", "sword", 5),
            Player(self, "heart", "shield", 6),
            Player(self, "heart", "sword", 7),
        ]

    def reset(self, player_idx=-1):
        self.player_idx=player_idx
        self.arrows = []
        self.points = [0,0,0,0]
        for player in self.players:
             player.reset()
        self.render()
        return self.get_observations(), {}

    def step(self, actions):
        for n, player in enumerate(self.players):
            player.step(actions[n])

        for n,arrow in list(enumerate(self.arrows))[::-1]:
            arrow.step()
            if arrow.team == -1:
                del self.arrows[n]

        for n, player in enumerate(self.players):
             if player.health <= 0:
                 player.reset()

        self.render()
        if self.render_mode != "None":
            pygame.event.pump()
            if self.render_mode == "god":
                pygame.transform.scale(self.surface, np.multiply(self.map_size, scale), self.screen)
                pygame.display.flip()
                self.clock.tick(15)

        rewards = [0 for i in range(len(self.players))]
        terminated = np.array([0 for i in range(len(self.players))])

        team_capture = [0,0,0,0]

        for n,player in enumerate(self.players):
            x,y = player.pos
            if 54<=x<=80 and 54<=y<=80:
                team_capture[team_idxs[player.team]] = 1

        for n,player in enumerate(self.players):
            r = 0
            x,y = player.pos
            if team_capture[team_idxs[player.team]]:
                r += 0.2
            #r -= sum(team_capture)/15
            #r += (min((1-math.dist(player.pos, (67,67))/50), 0.5)-0.2)/50
            r += player.r
            rewards[n] = r - 0.025

        for n,capture in enumerate(team_capture):
            self.points[n] += 0.2 * capture

        if max(self.points) >= 100:
            terminated += 1
            self.reset()
            
        return self.get_observations(), rewards, terminated, 0, {}

    def get_observations(self):
        observations = []
        for n,player in enumerate(self.players):
            observations.append(player.get_obs())
        return observations

    def render(self):
        """Render the game state"""
        self.surface.blit(background, (0,0))

        bar_width = 27
        p0 = self.points[0] / 100  
        p1 = self.points[1] / 100  
        p2 = self.points[2] / 100  
        p3 = self.points[3] / 100
        py_2 = 135 - bar_width
        px_2 = 135 - 2

        pygame.draw.rect(self.surface, colors["p0_bg"], (0, 0,       2, bar_width))
        pygame.draw.rect(self.surface, colors["p1_bg"], (px_2, 0,    2, bar_width))
        pygame.draw.rect(self.surface, colors["p2_bg"], (0, py_2,    2, bar_width))
        pygame.draw.rect(self.surface, colors["p3_bg"], (px_2, py_2, 2, bar_width))

        pygame.draw.rect(self.surface, colors["p0_fg"], (0, bar_width - int(bar_width * p0), 2, int(bar_width * p0)))
        pygame.draw.rect(self.surface, colors["p1_fg"], (px_2, bar_width - int(bar_width * p1), 2, int(bar_width * p1)))
        pygame.draw.rect(self.surface, colors["p2_fg"], (0, py_2+bar_width - int(bar_width * p2),    2, int(bar_width * p2)))
        pygame.draw.rect(self.surface, colors["p3_fg"], (px_2, py_2+bar_width - int(bar_width * p3), 2, int(bar_width * p3)))
    
        # Draw players
        for player in self.players:
            player.render()

        for arrow in self.arrows:
            arrow.render()

        self.surface_array = np.transpose(pygame.surfarray.array3d(self.surface), (2,1,0)) / 255
        self.padded_surface_array = np.zeros((3,135+64,135+64), dtype=np.float32)
        self.padded_surface_array[:, 32:135+32, 32:135+32] = self.surface_array


    def close(self):
        if self.render_mode != "None":
            pygame.quit()


def get_user_actions(env, player_idx):
    """Get user input for movement and attack"""
    actions = [1, 1, 0, 0, 0]  # Default action (stay still)
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_a]: actions[0] = 0
    if keys[pygame.K_d]: actions[0] = 2
    if keys[pygame.K_w]: actions[1] = 0
    if keys[pygame.K_s]: actions[1] = 2
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            actions[2] = 1

    mx, my = pygame.mouse.get_pos()
    mx //= scale
    my //= scale
    px, py = env.players[player_idx].pos
    dx, dy = mx - px, my - py
    angle = (round(math.degrees(math.atan2(dy, dx))/18)*18)%360

    actions[3] = round(angle/90)
    actions[4] = round((angle-actions[3]*90)/18)
    actions[3], actions[4] = actions[3]%4, (actions[4]-3)%5

    return actions

def get_random_actions():
    return [1, 1, 0, 0, 0]
    return (random.randint(0,2), random.randint(0,2), random.randint(0,1), random.randint(0,3), random.randint(0,4))

if __name__ == "__main__":
    env = CTCEnvironment(render_mode="god")
    env.reset()

    running = True
    c = 0
    total_r = 0
    while running:
        c += 1
        actions = [
            get_user_actions(env),
            get_random_actions(),
            get_random_actions(),
            get_random_actions(),
            get_random_actions(),
            get_random_actions(),
            get_random_actions(),
            get_random_actions()
        ]
        obs, rew, term, trun, info = env.step(actions)
        total_r += rew[0]
        #if c % 10 == 0: print(total_r)
        if term[0] == 1:
            running = False

    pygame.quit()






