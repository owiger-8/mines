import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
from collections import namedtuple, deque
from PIL import Image

CELL_SIZE_AT_ZOOM = 30 
FACE_TO_BOARD_X_OFFSET = 470
FACE_TO_BOARD_Y_OFFSET = 100


BOARD_WIDTH = 30
BOARD_HEIGHT = 16
NUM_EPISODES = 10000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
TAU = 0.005

CONFIDENCE_SETTING = 0.1

IMAGE_MAP = {
    'unopened.png': -1, 'empty.png': 0, '1.png': 1, '2.png': 2, '3.png': 3,
    '4.png': 4,
}
TEMPLATE_IMAGES = {name: Image.open(f'images/{name}') for name in IMAGE_MAP.keys()}
FACE_SMILE_IMG = Image.open('images/facesmile.png')
FACE_DEAD_IMG = Image.open('images/facedead.png')
FACE_WIN_IMG = Image.open('images/facewin.png')

class MinesweeperEnv:
    def __init__(self):
        self.board_region = None
        self.cell_size = CELL_SIZE_AT_ZOOM
        self.cell_coords = None
        self.last_state = None
        self.face_region = None

    def _locate_board(self):
        try:
            face_pos = pyautogui.locateOnScreen(FACE_SMILE_IMG, confidence=0.7)
            if face_pos is None:
                raise Exception("Could not find the smiley face...")
            self.face_region = (face_pos.left, face_pos.top, face_pos.width, face_pos.height)
            board_left = face_pos.left - FACE_TO_BOARD_X_OFFSET
            board_top = face_pos.top + FACE_TO_BOARD_Y_OFFSET
            board_w_pixels = int(BOARD_WIDTH * self.cell_size)
            board_h_pixels = int(BOARD_HEIGHT * self.cell_size)
            self.board_region = (int(board_left), int(board_top), board_w_pixels, board_h_pixels)
            self.cell_coords = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 2), dtype=int)
            for y in range(BOARD_HEIGHT):
                for x in range(BOARD_WIDTH):
                    self.cell_coords[y, x] = [board_left + x * self.cell_size, board_top + y * self.cell_size]
            print(f"Board located at: {self.board_region}")
            return True
        except pyautogui.ImageNotFoundException as e:
            print(f"Error locating board: {e}")
            return False

    def get_state(self):
        if self.board_region is None: return None
        screenshot = pyautogui.screenshot(region=self.board_region)
        state = np.full((BOARD_HEIGHT, BOARD_WIDTH), -2, dtype=np.float32)
        for img_name, value in IMAGE_MAP.items():
            template = TEMPLATE_IMAGES[img_name]
            try:
                locations = list(pyautogui.locateAll(template, screenshot, confidence=CONFIDENCE_SETTING))
                for loc in locations:
                    x = round(loc.left / self.cell_size)
                    y = round(loc.top / self.cell_size)
                    if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                        state[y, x] = value
            except pyautogui.ImageNotFoundException:
                continue
        self.last_state = state
        return torch.from_numpy(state).unsqueeze(0).unsqueeze(0)

    def reset(self):
        if self.board_region is None:
            if not self._locate_board():
                return None
        try:
            face_center = pyautogui.center(self.face_region)
            pyautogui.click(face_center)
            time.sleep(0.5)
            return self.get_state()
        except TypeError:
             print("Could not find smiley face to reset.")
        return None

    def step(self, action):
        y, x = np.unravel_index(action, (BOARD_HEIGHT, BOARD_WIDTH))
        cell_x_pixel = self.cell_coords[y, x, 0] + self.cell_size // 2
        cell_y_pixel = self.cell_coords[y, x, 1] + self.cell_size // 2
        pyautogui.click(cell_x_pixel, cell_y_pixel)
        time.sleep(0.05)
        try:
            if pyautogui.locateOnScreen(FACE_DEAD_IMG, region=self.face_region, confidence=CONFIDENCE_SETTING, grayscale=True):
                return self.get_state(), -100.0, True
            if pyautogui.locateOnScreen(FACE_WIN_IMG, region=self.face_region, confidence=CONFIDENCE_SETTING, grayscale=True):
                return self.get_state(), 200.0, True
        except pyautogui.ImageNotFoundException:
            pass
        reward = -1.0
        done = False
        new_state_tensor = self.get_state()
        if new_state_tensor is None:
            return self.last_state, -100.0, True
        return new_state_tensor, reward, done

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        def get_conv_output_size(size_h, size_w):
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, size_h, size_w)
                c1 = self.bn1(self.conv1(dummy_input))
                c2 = self.bn2(self.conv2(c1))
                return c2.flatten().shape[0]
        linear_input_size = get_conv_output_size(h, w)
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))

if __name__ == '__main__':
    env = MinesweeperEnv()
    print("Starting in 5 seconds...")
    print("Please make sure the Minesweeper game is fully visible on screen.")
    time.sleep(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_actions = BOARD_WIDTH * BOARD_HEIGHT
    policy_net = DQN(BOARD_HEIGHT, BOARD_WIDTH, n_actions).to(device)
    target_net = DQN(BOARD_HEIGHT, BOARD_WIDTH, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    steps_done = 0

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = policy_net(state)
                state_numpy = state.cpu().squeeze(0).squeeze(0).numpy()
                opened_mask = (state_numpy != -1).flatten()
                q_values[0, opened_mask] = -1e9
                return q_values.max(1)[1].view(1, 1)
        else:
            state_numpy = state.cpu().squeeze(0).squeeze(0).numpy()
            unopened_indices = np.where(state_numpy.flatten() == -1)[0]
            if len(unopened_indices) == 0: return torch.tensor([[0]], device=device, dtype=torch.long)
            action = random.choice(unopened_indices)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE: return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    
    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        if state is None:
            print("Failed to start game. Retrying...")
            time.sleep(2)
            continue
        state = state.to(device)
        center_y, center_x = BOARD_HEIGHT // 2, BOARD_WIDTH // 2
        first_action_index = center_y * BOARD_WIDTH + center_x
        first_action_tensor = torch.tensor([[first_action_index]], device=device, dtype=torch.long)
        next_state, reward, done = env.step(first_action_index)
        reward = torch.tensor([reward], device=device)
        if done: next_state = None
        elif next_state is not None: next_state = next_state.to(device)
        memory.push(state, first_action_tensor, next_state, reward)
        state = next_state
        episode_steps = 1
        if not done:
            for t in range(1, BOARD_WIDTH * BOARD_HEIGHT):
                episode_steps += 1
                action = select_action(state)
                next_state, reward, done = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                if done: next_state = None
                elif next_state is not None: next_state = next_state.to(device)
                memory.push(state, action, next_state, reward)
                state = next_state
                optimize_model()
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
                if done: break
                if state is None: break
        
        print(f"Episode {i_episode} finished after {episode_steps} steps. Reward: {reward.item()}")
    
    print('Training complete')
