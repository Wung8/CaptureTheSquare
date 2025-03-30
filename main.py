import torch
import cv2
import numpy as np
import pygame, time

from capture_the_cube_temp import CTCEnvironment as env
from capture_the_cube_temp import get_user_actions
from RL.Recurrent_PPO import Recurrent_PPO as PPO
from RL.type_aliases import LSTMStates

hidden_state_shape = (1, 96, 256)
n_agents = 8

def run(self, env, player_idx, n_steps=100_000, **kwargs):
    points = []
    cumulative_reward = 0
    obs, info = env.reset(player_idx)
    hidden_state_shape = list(self.hidden_state_shape)
    hidden_state_shape[1] = n_agents
    lstm_states = LSTMStates(
        (
            torch.zeros(hidden_state_shape, dtype=torch.float32),
            torch.zeros(hidden_state_shape, dtype=torch.float32),
        ),
        (
            torch.zeros(hidden_state_shape, dtype=torch.float32),
            torch.zeros(hidden_state_shape, dtype=torch.float32),
        )
    )
    episode_starts = [[0] for i in range(n_agents)]
    for step in range(n_steps):
        obs = np.array(obs, dtype=np.float32)
        with torch.no_grad():
            episode_starts = torch.tensor(episode_starts, dtype=torch.float32)
            action, log_probs, values, lstm_states = self.get_action(np.expand_dims(obs,0), lstm_states, episode_starts)
        action = action.numpy()
        if player_idx != -1:
            action[player_idx] = get_user_actions(env, player_idx)
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated[0]
        
        cumulative_reward += sum(reward)
        
        if done: break
        points = env.points
        obs = new_obs

    print(points)
    winner = ["diamond","club","spade","heart"][np.argmax(points)]
    font = pygame.font.Font(None, 50)
    text = f"{winner} wins!"
    text_color = colors[winner]
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(675 // 2, 675 // 2))
    
    for i in range(45):
        env.render()
        pygame.transform.scale(env.surface, np.multiply(env.map_size, 5), env.screen)
        env.screen.blit(text_surface, text_rect)

        pygame.display.flip()
        env.clock.tick(15)

    screen = pygame.display.set_mode((screen_x, screen_y))
        
    return cumulative_reward


w = 100
h = 50

# Window settings
screen_x, screen_y = 405, 540
screen = pygame.display.set_mode((screen_x, screen_y))
pygame.display.set_caption("Game Menu")

font = pygame.font.Font(None, 22)


colors = {
    "light_grey": (180, 180, 180),
    "grey": (130, 130, 130),
    
    "button_bg": (50, 50, 50),
    "button_hover": (80, 80, 80),
    "button_text": (255, 255, 255),

    "diamond": (89, 113, 179),
    "club": (78, 142, 72),
    "spade": (113, 88, 149),
    "heart": (162, 70, 70),
}

player_buttons = [
    ("Diamond 1\n(bow)", pygame.Rect(50, 100, w, h)),
    ("Diamond 2\n(sword)", pygame.Rect(50, 180, w, h)),
    ("Club 1\n(bow)",      pygame.Rect(255, 100, w, h)),
    ("Club 2\n(bow)",      pygame.Rect(255, 180, w, h)),
    ("Spade 1\n(sword)",     pygame.Rect(50, 260, w, h)),
    ("Spade 2\n(sword)",  pygame.Rect(50, 340, w, h)),
    ("Heart 1\n(shield)",   pygame.Rect(255, 260, w, h)),
    ("Heart 2\n(sword)",  pygame.Rect(255, 340, w, h)),
]

def show_cover():
    cover_image = pygame.image.load("cover.png")  # Load the cover image
    cover_image = pygame.transform.scale(cover_image, (cover_image.get_width() * 3, cover_image.get_height() * 3))  # Scale up by 3x
    
    start_time = time.time()  # Get the current time
    while time.time() - start_time < 3:  # Loop for 3 seconds
        screen.fill((0, 0, 0))  # Clear screen (optional)
        screen.blit(cover_image, (0, 0))  # Draw the scaled cover image
        pygame.display.flip()  # Update the display
        pygame.event.pump()  # Process events to prevent freezing

def draw_button(text, rect, hovered):
    """Draw a button with multi-line text."""
    color = colors["button_hover"] if hovered else colors["button_bg"]
    pygame.draw.rect(screen, color, rect)

    # Split text into two lines
    lines = text.split("\n")
    
    # Render each line separately
    text_surfaces = [font.render(line, True, colors["button_text"]) for line in lines]
    
    # Stack text lines inside the button
    total_height = sum(surf.get_height() for surf in text_surfaces)
    start_y = rect.centery - total_height // 2

    for text_surf in text_surfaces:
        text_rect = text_surf.get_rect(center=(rect.centerx, start_y))
        screen.blit(text_surf, text_rect)
        start_y += text_surf.get_height()  # Move down for next line

def main_menu():
    """Display the main menu and return selected game mode."""
    h = 40
    w = 200
    button_0p = pygame.Rect(int(screen_x / 2 - w / 2), 100, w, h)
    button_1p = pygame.Rect(int(screen_x / 2 - w / 2), 170, w, h)
    
    running = True
    while running:
        screen.fill(colors["light_grey"])
        
        # Mouse position
        mx, my = pygame.mouse.get_pos()
        
        # Check if hovering over buttons
        hover_0p = button_0p.collidepoint(mx, my)
        hover_1p = button_1p.collidepoint(mx, my)
        
        # Draw buttons
        draw_button("0 Players (AI Mode)", button_0p, hover_0p)
        draw_button("1 Player (Select Player)", button_1p, hover_1p)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_0p.collidepoint(mx, my):
                    return "0p", -1
                if button_1p.collidepoint(mx, my):
                    return "1p", player_selection()

        pygame.display.flip()

def player_selection():
    """Let the player select a character before starting the game."""

    selected_player = None

    while selected_player is None:
        screen.fill(colors["light_grey"])

        # Mouse position
        mx, my = pygame.mouse.get_pos()

        # Draw buttons
        for text, rect in player_buttons:
            hovered = rect.collidepoint(mx, my)
            draw_button(text, rect, hovered)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, (text, rect) in enumerate(player_buttons):
                    if rect.collidepoint(mx, my):
                        return i  # Player is selected

        pygame.display.flip()
        

if __name__ == "__main__":
    show_cover()
    while True:
        pygame.display.set_mode((screen_x, screen_y))
        game_mode, player_idx = main_menu()

        agent = PPO(
            env=None,
            observation_space=(3,48,48),
            action_space=(3,3,2,4,5),
            n_steps=0,
        )
        agent.model = torch.load("model.pt")

        run(agent, env(render_mode="god"), player_idx)

    
