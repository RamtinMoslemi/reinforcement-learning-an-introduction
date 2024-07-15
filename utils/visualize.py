import gymnasium as gym
import numpy as np
import cv2
from PIL import Image


def greedy(q_value, state):
    return np.argmax(q_value[state])


def show_policy(env, q_value, act=greedy):
    state, info = env.reset()
    while True:
        action = act(q_value, state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


def make_mp4(env, q_value, name='file_name', act=greedy):
    assert env.render_mode == 'rgb_array'
    frames = list()
    state, info = env.reset()
    while True:
        frames.append(env.render())
        action = act(q_value, state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break
    env.close()
    height, width, layers = frames[0].shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4, 'XVID' for .avi
    video = cv2.VideoWriter(name + '.mp4', fourcc, env.metadata['render_fps'], (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()


def make_gif(env, q_value, name='file_name', act=greedy):
    assert env.render_mode == 'rgb_array'
    frames = []
    state, info = env.reset()
    while True:
        frame = env.render()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV
        action = act(q_value, state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frame = env.render()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV
            break
    env.close()

    # Convert frames to PIL images
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    # Save as GIF
    gif_path = f"{name}.gif"
    duration = 1000 // env.metadata['render_fps']
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:], loop=0, duration=duration
    )
