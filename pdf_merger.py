import fitz  # PyMuPDF
import os
from PIL import Image, ImageDraw, ImageFont
import io

def merge_pdfs_to_grid_simple(pdf_files, output_path, grid_rows=3, grid_cols=3):
    """
    使用PyMuPDF直接合并PDF到网格布局
    """
    # 创建新的PDF文档
    output_doc = fitz.open()
    
    # 计算页面大小 (使用更大的页面)
    page_width = 2000  # A4宽度的3倍
    page_height = 2000  # A4高度的3倍
    
    # 创建一个新页面
    page = output_doc.new_page(width=page_width, height=page_height)
    
    # 计算每个单元格的尺寸
    cell_width = page_width / grid_cols
    cell_height = page_height / grid_rows
    
    # 大幅减少左右边距，保持上下边距
    margin_x = 50 # 左右边距
    margin_y = 10  # 上下边距
    title_height = 50
    
    # 算法名称映射
    algorithm_mapping = {
        'beam': 'BS-PRM',
        'delta': '  -PRM', 
        'spars': 'SPARS2'
    }
    
    # 环境名称映射
    environment_mapping = {
        'indoor': 'Indoor',
        'maze': 'Maze',
        'random': 'Cluttered'
    }
    
    for i, pdf_file in enumerate(pdf_files):
        if not os.path.exists(pdf_file):
            print(f"警告: 文件 {pdf_file} 不存在，跳过")
            continue
        
        # 计算网格位置
        row = i // grid_cols
        col = i % grid_cols
        
        # 计算单元格位置
        x_start = col * cell_width
        y_start = row * cell_height
        
        # 从文件名提取并映射标题
        filename = os.path.basename(pdf_file)
        parts = filename.replace('.pdf', '').split('_')
        
        if len(parts) >= 2:
            algorithm = parts[0].lower()
            environment = parts[1].lower()
            
            # 使用映射获取显示名称
            alg_display = algorithm_mapping.get(algorithm, algorithm.upper())
            env_display = environment_mapping.get(environment, environment.capitalize())
            title = f"{alg_display} {env_display}"
        else:
            title = filename.replace('.pdf', '').upper()
        
        try:
            # 打开PDF文件
            doc = fitz.open(pdf_file)
            src_page = doc.load_page(0)  # 取第一页
            
            # 计算源页面的尺寸
            src_rect = src_page.rect
            
            # 计算目标矩形（大幅减少左右边距）
            target_width = cell_width - 2 * margin_x
            target_height = cell_height - 2 * margin_y - title_height
            
            # 计算缩放比例
            scale_x = target_width / src_rect.width
            scale_y = target_height / src_rect.height
            scale = min(scale_x, scale_y)
            
            # 计算实际大小和位置
            scaled_width = src_rect.width * scale
            scaled_height = src_rect.height * scale
            
            # 居中位置（左右居中，图片占用更多空间）
            img_x = x_start + margin_x + (target_width - scaled_width) / 2
            img_y = y_start + margin_y + title_height
            
            # 创建目标矩形
            target_rect = fitz.Rect(img_x, img_y, 
                                   img_x + scaled_width, 
                                   img_y + scaled_height)
            
            # 将源页面插入到目标页面
            page.show_pdf_page(target_rect, doc, 0)
            
            # 计算标题的居中位置（与整个单元格中心对齐）
            title_x = x_start + cell_width / 2
            title_y = y_start + margin_y + 25
            
            # 获取文本尺寸以确保居中
            # 使用textlength方法获取文本宽度
            text_width = fitz.get_text_length(title, fontname="tibo", fontsize=30)
            title_x_centered = title_x - text_width / 2
            
            title_point = fitz.Point(title_x_centered, title_y)
            
            # 只插入一次标题文本
            page.insert_text(title_point, title, 
                           fontsize=30, 
                           fontname="tibo",
                           color=(0, 0, 0))
            
            # 绘制单元格边框（可选）
            # border_rect = fitz.Rect(x_start, y_start, 
            #                        x_start + cell_width, 
            #                        y_start + cell_height)
            # page.draw_rect(border_rect, color=(0.8, 0.8, 0.8), width=0.5)
            
            doc.close()
            
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {e}")
            continue
    
    # 保存输出文档
    output_doc.save(output_path)
    output_doc.close()
    print(f"网格PDF已保存到: {output_path}")

def main():
    # 您的PDF文件列表
    pdf_files = [
        "pursuer_strategies/PRM/results/random/beam_random_520.pdf",
        "pursuer_strategies/PRM/results/random/delta_random_520.pdf", 
        "pursuer_strategies/PRM/results/random/spars_random_520.pdf",
        "pursuer_strategies/PRM/results/maze/beam_maze_114.pdf",
        "pursuer_strategies/PRM/results/maze/delta_maze_114.pdf",
        "pursuer_strategies/PRM/results/maze/spars_maze_114.pdf",
        "pursuer_strategies/PRM/results/indoor/beam_indoor.pdf",
        "pursuer_strategies/PRM/results/indoor/delta_indoor.pdf",
        "pursuer_strategies/PRM/results/indoor/spars_indoor.pdf"
    ]
    
    # 输出文件路径
    output_path = "pursuer_strategies/PRM/results/charts/grid_analysis_3x3.pdf"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 合并为3x3网格
    merge_pdfs_to_grid_simple(pdf_files, output_path, grid_rows=3, grid_cols=3)

if __name__ == "__main__":
    main()