import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 导入自定义模块
from data_generator import PathDataGenerator, PathPlanningDataset
from encoder_decoder import OneShotPathPlanner
from config import ENV_CONFIG

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 自定义收集函数，处理批次中不同长度的数据
def custom_collate_fn(batch):
    """
    自定义收集函数，处理不同长度的路径
    """
    # 提取一批中的所有键
    keys = batch[0].keys()
    
    result = {}
    for key in keys:
        if key in ['input', 'target']:  # 张量数据，可以直接堆叠
            result[key] = torch.stack([sample[key] for sample in batch])
        elif key in ['start', 'goal']:  # 坐标数据，转换为列表
            result[key] = [sample[key] for sample in batch]
        elif key == 'path':  # 路径数据，保持为列表格式
            result[key] = [sample[key] for sample in batch]
        else:  # 其他数据，保持原样
            result[key] = [sample[key] for sample in batch]
    
    return result
        
# 带权重的扁平化二值交叉熵损失
class WeightedFlattenedBCELoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super(WeightedFlattenedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # 扁平化输入和目标
        batch_size = inputs.size(0)
        inputs_flat = inputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # 计算带权重的BCE损失
        # loss = -(w_p * y * log(x) + (1-y) * log(1-x))
        eps = 1e-7  # 防止log(0)
        loss_pos = targets_flat * torch.log(inputs_flat + eps)
        loss_neg = (1 - targets_flat) * torch.log(1 - inputs_flat + eps)
        
        # 对正样本使用更高权重
        weighted_loss = -1.0 * (self.pos_weight * loss_pos + loss_neg)
        
        # 计算每个样本的平均损失
        return weighted_loss.mean()

# 带权重的组合损失函数
class WeightedCombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.6, dice_weight=0.4, pos_weight=10.0):
        super(WeightedCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = WeightedFlattenedBCELoss(pos_weight=pos_weight)
        self.dice_loss = FlattenedDiceLoss()  # Dice损失天然处理类不平衡
        
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice
# 扁平化的Dice损失
class FlattenedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(FlattenedDiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # 扁平化输入和目标，但保留批次维度
        batch_size = inputs.size(0)
        inputs_flat = inputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # 计算每个样本的Dice系数
        intersection = (inputs_flat * targets_flat).sum(1)
        union = inputs_flat.sum(1) + targets_flat.sum(1)
        
        # Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回Dice损失 (1 - Dice系数的平均值)
        return 1 - dice.mean()



def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_dir):
    """训练函数"""
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 用于记录训练和验证损失
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 当前时间，用于生成唯一的模型ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"pathplan_{timestamp}"
    
    # 训练循环
    print("开始训练模型...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累加损失
            epoch_train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 累加损失
                epoch_val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 打印每个epoch的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'训练损失: {avg_train_loss:.4f}, '
              f'验证损失: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_encoder_decoder(model, model_dir, model_id)  # 保存编码器和解码器
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_id}_final.pth"))
    print("训练完成")
    
    return model_id


def evaluate(model, test_loader, device):
    """评估函数"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            
            # 收集所有输出和目标以计算指标
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合并批次数据
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 将输出和目标展平为1D数组以计算指标
    flat_outputs = all_outputs.reshape(-1)
    flat_targets = all_targets.reshape(-1)
    
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(flat_targets, flat_outputs)
    average_precision = average_precision_score(flat_targets, flat_outputs)
    print(flat_targets, flat_outputs)
    # 计算最佳阈值（F1分数最高）
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"平均精确率: {average_precision:.4f}")
    print(f"最佳阈值: {best_threshold:.4f}")
    
    # 使用最佳阈值二值化预测
    binary_outputs = (flat_outputs > best_threshold).astype(np.int32)
    
    # 计算准确率
    accuracy = np.mean(binary_outputs == flat_targets)
    print(f"准确率: {accuracy:.4f}")
    
    return {
        'average_precision': average_precision,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
    }


def visualize_with_pygame(model, test_loader, device, threshold=0, num_samples=5):
    """使用pygame可视化模型预测结果"""
    import pygame
    import time
    
    # 初始化pygame
    pygame.init()
    
    # 定义尺寸和颜色
    cell_size = 20  # 每个格子的像素尺寸
    grid_width = 30
    grid_height = 15
    win_width = grid_width * cell_size
    win_height = grid_height * cell_size
    
    # 定义颜色
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)
    
    # 创建窗口
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption("路径规划模型预测可视化")
    
    # 设置字体
    font = pygame.font.SysFont('arial', 16)
    
    model.eval()
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if count >= num_samples:
                break
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # 获取模型预测
            outputs = model(inputs)
            
            # 处理批次中的样本
            for i in range(min(inputs.size(0), num_samples - count)):
                # 获取数据
                obstacle_map = inputs[i, 2].cpu().numpy()  # 障碍物图
                start_map = inputs[i, 0].cpu().numpy()     # 起点图
                goal_map = inputs[i, 1].cpu().numpy()      # 终点图
                true_path = targets[i, 0].cpu().numpy()    # 真实路径
                pred_path = outputs[i, 0].cpu().numpy()    # 预测路径

                # 获取起点和终点坐标
                start_points = np.argwhere(start_map > 0.5)
                goal_points = np.argwhere(goal_map > 0.5)
                goal_pos = (int(goal_points[0][1]), int(goal_points[0][0]))
                if len(start_points) > 0 and len(goal_points) > 0:
                    start_pos = [(int(x), int(y)) for y, x in start_points]

                else:
                    start_pos = [(0, 0)]
                    goal_pos = [(grid_width-1, grid_height-1)]
                
                # 使用阈值二值化预测路径
                binary_pred = (pred_path > threshold).astype(int)
                
                # 清除屏幕
                screen.fill(WHITE)
               

                # 绘制格子
                for y in range(grid_height):
                    for x in range(grid_width):
                        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                        small_rect = pygame.Rect(x * cell_size + 5, y * cell_size + 5,
                                                cell_size - 10, cell_size - 10)
                        # 绘制障碍物
                        if obstacle_map[y, x] > 0.5:
                            pygame.draw.rect(screen, BLACK, rect)
                        
                        # 绘制真实路径
                        if true_path[y, x] > 0.5:
                            pygame.draw.rect(screen, GREEN, rect,2)
                        
                        # 绘制预测路径
                        if binary_pred[y, x] > 0:
                            pygame.draw.rect(screen, PURPLE, small_rect)
                        
                        # 绘制格子边界
                        pygame.draw.rect(screen, (200, 200, 200), rect, 1)
                
                # 绘制起点和终点
                for start in start_pos:
                    start_rect = pygame.Rect(start[0] * cell_size, start[1] * cell_size, 
                                             cell_size, cell_size)
                    pygame.draw.rect(screen, RED, start_rect)
                goal_rect = pygame.Rect(goal_pos[0] * cell_size, goal_pos[1] * cell_size, 
                                       cell_size, cell_size)
                pygame.draw.rect(screen, RED, start_rect)
                pygame.draw.rect(screen, BLUE, goal_rect)
                
                # 绘制说明文本
                info_text = f"样本 {count+1}/{num_samples} (阈值={threshold:.2f})"
                text_surface = font.render(info_text, True, BLACK)
                screen.blit(text_surface, (10, 10))
                
                legend1 = font.render("green: true path", True, GREEN)
                legend2 = font.render("purple: predicted path", True, PURPLE)
                legend3 = font.render("red: start point", True, RED)
                legend4 = font.render("blue: goal point", True, BLUE)
                legend5 = font.render("black: obstacle", True, BLACK)
                
                screen.blit(legend1, (win_width - 120, 10))
                screen.blit(legend2, (win_width - 120, 30))
                screen.blit(legend3, (win_width - 120, 50))
                screen.blit(legend4, (win_width - 120, 70))
                screen.blit(legend5, (win_width - 120, 90))

                # 更新显示
                pygame.display.flip()
                
                # 等待用户按键继续
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        if event.type == pygame.KEYDOWN:
                            waiting = False
                
                count += 1
                if count >= num_samples:
                    break
    
    pygame.quit()
    print("可视化完成!")

def save_encoder_decoder(model, model_dir, model_id):
    """分别保存编码器和解码器"""
    #first delete previous saved models

    encoder_path = os.path.join(model_dir, f"{model_id}_encoder.pth")
    decoder_path = os.path.join(model_dir, f"{model_id}_decoder.pth")

    if os.path.exists(encoder_path):
        os.remove(encoder_path)
    if os.path.exists(decoder_path):
        os.remove(decoder_path)

    # 保存编码器
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"编码器已保存到: {encoder_path}")
    
    # 保存解码器
    torch.save(model.decoder.state_dict(), decoder_path)
    print(f"解码器已保存到: {decoder_path}")

def load_encoder_decoder(model, model_dir, model_id):
    """分别加载编码器和解码器"""
    encoder_path = os.path.join(model_dir, f"{model_id}_encoder.pth")
    decoder_path = os.path.join(model_dir, f"{model_id}_decoder.pth")
    
    # 加载编码器
    model.encoder.load_state_dict(torch.load(encoder_path))
    print(f"编码器已加载: {encoder_path}")
    
    # 加载解码器
    model.decoder.load_state_dict(torch.load(decoder_path))
    print(f"解码器已加载: {decoder_path}")

def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 配置参数
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    num_samples = 120000 # 生成的总样本数
    model_dir = "./utils/env_encoder/model"
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 生成数据集
    print("生成路径规划数据集...")
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height = ENV_CONFIG['gridnum_height']
    generator = PathDataGenerator(grid_width, grid_height)
    dataset_data = generator.generate_dataset(num_samples)
    dataset = PathPlanningDataset(dataset_data)
    
    # 划分数据集
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器 - 添加自定义的collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           num_workers=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            num_workers=4, collate_fn=custom_collate_fn)
    
    # 初始化模型
    model = OneShotPathPlanner(input_channels=3, output_channels=1)

    model = model.to(device)
    
    # 损失函数和优化器 - 使用扁平化的损失函数
        # 修改为带权重的损失函数
    criterion = WeightedCombinedLoss(
        bce_weight=0.6, 
        dice_weight=0.4, 
        pos_weight=15.0  # 可根据实际正负样本比例调整
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    model_id = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_dir)
    
    # 评估模型
    print("测试模型...")
    load_encoder_decoder(model, model_dir, model_id)
    metrics = evaluate(model, test_loader, device)
    
    # 使用Pygame可视化结果
    print("使用Pygame可视化预测结果...")
    threshold = metrics['best_threshold']  # 使用评估中找到的最佳阈值
    visualize_with_pygame(model, test_loader, device, threshold=threshold, num_samples=5)
    
    print("训练和评估完成!")
    
def direct_evaluate():
    #select the newest model and geneate test data
    model_dir = "./utils/env_encoder/model"
    encoder_model_files = [f for f in os.listdir(model_dir) if f.endswith('encoder.pth')]
    if not encoder_model_files:
        print("没有找到最佳模型文件")
        return
    encoder_model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_encoder_model_file = encoder_model_files[0]
    encoder_model_path = os.path.join(model_dir, latest_encoder_model_file)
    encoder_model_id = latest_encoder_model_file.split('_')[0]

    decoder_model_files = [f for f in os.listdir(model_dir) if f.endswith('decoder.pth')]
    if not decoder_model_files:
        print("没有找到最佳解码器模型文件")
        return
    decoder_model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_decoder_model_file = decoder_model_files[0]
    decoder_model_path = os.path.join(model_dir, latest_decoder_model_file)
    decoder_model_id = latest_decoder_model_file.split('_')[0]


    #generate test data
    generator = PathDataGenerator(ENV_CONFIG['gridnum_width'], ENV_CONFIG['gridnum_height'])
    test_data = generator.generate_dataset(100)  # 生成1000个测试样本
    test_dataset = PathPlanningDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, collate_fn=custom_collate_fn)    


    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneShotPathPlanner(input_channels=3, output_channels=1)
    # 加载编码器
    model.encoder.load_state_dict(torch.load(encoder_model_path))
    print(f"编码器已加载: {encoder_model_path}")
    
    # 加载解码器
    model.decoder.load_state_dict(torch.load(decoder_model_path))
    print(f"解码器已加载: {decoder_model_path}")
    model = model.to(device)
    #evaluate model
    metrics = evaluate(model, test_loader, device)
    print(f"模型评估结果: {metrics}")
    #可视化结果
    visualize_with_pygame(model, test_loader, device, threshold=metrics['best_threshold'], num_samples=5)
    print("评估和可视化完成!")  

if __name__ == "__main__":
    #main()
    direct_evaluate()