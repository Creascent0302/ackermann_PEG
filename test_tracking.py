"""
before testing, switch the policy to astar in strategy_selector.py
"""
import time
from env import GameENV
from strategy_selector import pursuer_strategy_selector, evader_strategy_selector
from render import PygameRenderer
import pygame  # 添加pygame以监听事件
import random


def main():
    """主函数"""
    # 初始化环境
    seed = random.randint(0, 10000)  # 随机种子
    seed = 8804# 固定种子以便调试
    num_pursuers = 1  # 追捕者数量
    print(f"使用随机种子: {seed}")  # 打印随机种子以便调试
    env = GameENV(num_pursuers,seed)
    observe = env.reset()
    pursuer_selector = pursuer_strategy_selector()
    evader_selector = evader_strategy_selector()
    # 初始化pygame渲染器
    renderer = PygameRenderer(env.gridnum_width, env.gridnum_height)

    print("=== 追逃游戏开始 ===")
    
    step_count = 0
    running = True
    
    # 主仿真循环
    while running and not env.done and step_count < env.max_steps:
        # 监听关闭窗口事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        
        if not running:
            break

        # 获取动作
        evader_action = (0,0)
        evader_path = []
        preview_point = (0,0)
        pursuers_actions = []
        
        pursuers_actions,  predicted_evader_pos, preview_point = pursuer_selector.select_strategy(observe)
        # 执行动作
        actions = {
            'evader': evader_action,
            'pursuers': pursuers_actions
        }

        observe, reward, done, info = env.step(actions)
        # 渲染
        running = renderer.render(env, step_count, predicted_evader_pos, evader_path, preview_point)
        step_count += 1
        #time.sleep(0.1)  # 控制速度
    
    # 游戏结束
    print(f"\n游戏结束! 总步数: {step_count}")
    if env.done  and (env.evader_collision or env.capture):
        print("追捕者获胜!")
    elif step_count == env.max_steps or env.pursuers_collision:
        print("逃避者获胜!")
    else:
        print("游戏被手动终止!")
    
    # 等待几秒后关闭
    time.sleep(1)
    renderer.close()

if __name__ == "__main__":
    for i in range(15):
        print(f"=== 试验 {i + 1} ===")
        main()