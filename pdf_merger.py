import fitz  # PyMuPDF
import os

def merge_pdfs_to_grid_simple(pdf_files, output_path, grid_rows=3, grid_cols=4):
    """
    使用PyMuPDF直接合并PDF到网格布局，标题放置在下方并导出为SVG
    """
    # 创建新的PDF文档作为中转
    output_doc = fitz.open()
    
    # 计算页面大小 (4列所以相应加宽，保持比例协调)
    page_width = 2666  # A4宽度的4倍左右
    page_height = 2000  # A4高度的3倍
    
    # 创建一个新页面
    page = output_doc.new_page(width=page_width, height=page_height)
    
    # 计算每个单元格的尺寸
    cell_width = page_width / grid_cols
    cell_height = page_height / grid_rows
    
    # 调整边距：图片尽量撑满，把空间留给底部的标题
    margin_x = 30  # 左右边距
    margin_top = 20  # 顶部边距(减小)
    margin_bottom = 70  # 底部留给标题的空间
    
    # 算法名称映射 (增加 GSRM)
    algorithm_mapping = {
        'beam': 'BS-PRM',
        'delta': '  -PRM', 
        'spars': 'SPARS2',
        'gsrm': 'GSRM'
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
        
        # 计算单元格基准位置
        x_start = col * cell_width
        y_start = row * cell_height
        
        # 从文件名提取并映射标题
        filename = os.path.basename(pdf_file)
        parts = filename.replace('.pdf', '').split('_')
        
        # 【核心修改区】：矩阵式下标 (a1)...(c4)
        row_char = chr(97 + row)  # 97是'a'的ASCII码，0->a, 1->b, 2->c
        col_num = col + 1         # 0->1, 1->2, 2->3, 3->4
        label = f"{row_char}{col_num}"  # 组合成 a1, b2 等
        
        if len(parts) >= 2:
            algorithm = parts[0].lower()
            environment = parts[1].lower()
            
            alg_display = algorithm_mapping.get(algorithm, algorithm.upper())
            env_display = environment_mapping.get(environment, environment.capitalize())
            
            title = f"({label}) {alg_display} {env_display}"
        else:
            title = f"({label}) {filename.replace('.pdf', '').upper()}"
        
        try:
            # 打开源PDF文件
            doc = fitz.open(pdf_file)
            src_page = doc.load_page(0)
            
            # 计算源页面的尺寸
            src_rect = src_page.rect
            
            # 计算目标图片的最大可用矩形区域（留出底部标题空间）
            target_width = cell_width - 2 * margin_x
            target_height = cell_height - margin_top - margin_bottom
            
            # 计算缩放比例，保持原有宽高比
            scale_x = target_width / src_rect.width
            scale_y = target_height / src_rect.height
            scale = min(scale_x, scale_y)
            
            scaled_width = src_rect.width * scale
            scaled_height = src_rect.height * scale
            
            # 居中放置图片 (水平居中，垂直靠上)
            img_x = x_start + margin_x + (target_width - scaled_width) / 2
            img_y = y_start + margin_top
            
            # 创建目标矩形并插入PDF页面
            target_rect = fitz.Rect(img_x, img_y, img_x + scaled_width, img_y + scaled_height)
            page.show_pdf_page(target_rect, doc, 0)
            
            # --- 绘制底部标题 ---
            # X轴与当前单元格居中对齐
            title_x = x_start + cell_width / 2
            # Y轴放置在图片的正下方 (加上一点间隙)
            title_y = img_y + scaled_height + 40 
            
            # 获取文本宽度以确保严格居中
            text_width = fitz.get_text_length(title, fontname="tibo", fontsize=32)
            title_x_centered = title_x - text_width / 2
            
            title_point = fitz.Point(title_x_centered, title_y)
            
            # 插入带序号的标题
            page.insert_text(title_point, title, 
                           fontsize=32, 
                           fontname="tibo",
                           color=(0, 0, 0))
            
            doc.close()
            
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {e}")
            continue
    
    # 提取整页的 SVG 矢量内容并写入文件
    print("正在生成高清 SVG 矢量图...")
    svg_content = page.get_svg_image(matrix=fitz.Identity)
    
    with open(output_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(svg_content)
        
    output_doc.close()
    print(f"✅ 大图合成成功，已保存至: {output_path}")

def main():
    # PDF文件列表 (3行4列 = 12个文件)
    pdf_files = [
        # 第一行：Cluttered (a1-a4)
        "pursuer_strategies/PRM/results/random/beam_random_520.pdf",
        "pursuer_strategies/PRM/results/random/delta_random_520.pdf", 
        "pursuer_strategies/PRM/results/random/spars_random_520.pdf",
        "pursuer_strategies/PRM/results/random/gsrm_random.pdf",      
        
        # 第二行：Maze (b1-b4)
        "pursuer_strategies/PRM/results/maze/beam_maze_114.pdf",
        "pursuer_strategies/PRM/results/maze/delta_maze_114.pdf",
        "pursuer_strategies/PRM/results/maze/spars_maze_114.pdf",
        "pursuer_strategies/PRM/results/maze/gsrm_maze.pdf",          
        
        # 第三行：Indoor (c1-c4)
        "pursuer_strategies/PRM/results/indoor/beam_indoor.pdf",
        "pursuer_strategies/PRM/results/indoor/delta_indoor.pdf",
        "pursuer_strategies/PRM/results/indoor/spars_indoor.pdf",
        "pursuer_strategies/PRM/results/indoor/gsrm_indoor.pdf"       
    ]
    
    output_path = "pursuer_strategies/PRM/results/charts/grid_analysis_3x4.svg"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 合并为3x4网格
    merge_pdfs_to_grid_simple(pdf_files, output_path, grid_rows=3, grid_cols=4)

if __name__ == "__main__":
    main()