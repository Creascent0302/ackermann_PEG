
import fitz  # PyMuPDF
import os


def merge_pdfs_to_grid_2x2(pdf_files, titles, output_path, fmt="pdf"):
    fmt = fmt.strip().lower()
    if fmt not in ("pdf", "svg"):
        raise ValueError(f"不支持的格式: {fmt!r}，请传入 'pdf' 或 'svg'")

    base, _ = os.path.splitext(output_path)
    output_path = f"{base}.{fmt}"

    output_doc = fitz.open()

    page_width  = 1600
    page_height = 1600

    page = output_doc.new_page(width=page_width, height=page_height)

    grid_rows = 2
    grid_cols = 2
    cell_width  = page_width  / grid_cols
    cell_height = page_height / grid_rows

    margin_outer  = 0   # 整张大图左右两侧外边距
    margin_inner  = -30    # ← 左右两列之间的间距，0 = 完全贴合
    margin_top    = 20
    margin_bottom = 70

    for i, (pdf_file, title) in enumerate(zip(pdf_files, titles)):
        if not os.path.exists(pdf_file):
            print(f"警告: 文件 {pdf_file} 不存在，跳过")
            continue

        row = i // grid_cols
        col = i % grid_cols

        x_start = col * cell_width
        y_start = row * cell_height

        margin_left  = margin_outer if col == 0             else margin_inner
        margin_right = margin_outer if col == grid_cols - 1 else margin_inner

        target_width  = cell_width  - margin_left - margin_right
        target_height = cell_height - margin_top  - margin_bottom

        try:
            doc      = fitz.open(pdf_file)
            src_page = doc.load_page(0)
            src_rect = src_page.rect

            scale         = min(target_width  / src_rect.width,
                                target_height / src_rect.height)
            scaled_width  = src_rect.width  * scale
            scaled_height = src_rect.height * scale

            img_x = x_start + margin_left + (target_width - scaled_width) / 2
            img_y = y_start + margin_top

            target_rect = fitz.Rect(img_x, img_y,
                                    img_x + scaled_width,
                                    img_y + scaled_height)
            page.show_pdf_page(target_rect, doc, 0)

            title_center_x   = x_start + margin_left + target_width / 2
            title_y          = img_y + scaled_height + 45
            text_width       = fitz.get_text_length(title, fontname="tibo", fontsize=32)
            title_x_centered = title_center_x - text_width / 2

            page.insert_text(fitz.Point(title_x_centered, title_y),
                             title,
                             fontsize=32,
                             fontname="tibo",
                             color=(0, 0, 0))
            doc.close()

        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {e}")
            continue

    if fmt == "svg":
        svg_content = page.get_svg_image(matrix=fitz.Identity)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    else:
        output_doc.save(output_path, garbage=4, deflate=True)

    output_doc.close()
    print(f"✅ 已保存 [{fmt.upper()}]: {output_path}")


def main():
    P = "pursuer_strategies/PRM/results"

    OUTPUT_FORMAT = "svg"

    fig1_files = [
        f"{P}/random/beam_random_520.pdf",
        f"{P}/maze/beam_maze_114.pdf",
        f"{P}/indoor/beam_indoor.pdf",
        f"{P}/four_rooms/beam_four_rooms_43.pdf",
    ]
    fig1_titles = [
        "(a1) BS-PRM  Cluttered",
        "(a2) BS-PRM  Maze",
        "(b1) BS-PRM  Indoor",
        "(b2) BS-PRM  Four Rooms",
    ]

    fig2_files = [
        f"{P}/random/beam_random_520.pdf",
        f"{P}/random/delta_random_520.pdf",
        f"{P}/random/spars_random_520.pdf",
        f"{P}/random/gsrm_random.pdf",
    ]
    fig2_titles = [
        "(a1) BS-PRM Cluttered",
        "(a2)   -PRM Cluttered",
        "(b1) SPARS2 Cluttered",
        "(b2) GSRM  Cluttered",
    ]

    fig3_files = [
        f"{P}/maze/beam_maze_114.pdf",
        f"{P}/maze/delta_maze_114.pdf",
        f"{P}/maze/spars_maze_114.pdf",
        f"{P}/maze/gsrm_maze.pdf",
    ]
    fig3_titles = [
        "(a1) BS-PRM Maze",
        "(a2)   -PRM Maze",
        "(b1) SPARS2 Maze",
        "(b2) GSRM Maze",
    ]

    fig4_files = [
        f"{P}/indoor/beam_indoor.pdf",
        f"{P}/indoor/delta_indoor.pdf",
        f"{P}/indoor/spars_indoor.pdf",
        f"{P}/indoor/gsrm_indoor.pdf",
    ]
    fig4_titles = [
        "(a1) BS-PRM Indoor",
        "(a2)   -PRM Indoor",
        "(b1) SPARS2 Indoor",
        "(b2) GSRM Indoor",
    ]

    out_dir = f"{P}/charts"
    os.makedirs(out_dir, exist_ok=True)

    tasks = [
        (fig1_files, fig1_titles, f"{out_dir}/fig1_beam_all_envs.pdf"),
        (fig2_files, fig2_titles, f"{out_dir}/fig2_random_all_algos.pdf"),
        (fig3_files, fig3_titles, f"{out_dir}/fig3_maze_all_algos.pdf"),
        (fig4_files, fig4_titles, f"{out_dir}/fig4_indoor_all_algos.pdf"),
    ]

    for files, titles, out_path in tasks:
        print(f"\n📌 生成: {os.path.basename(out_path)}")
        merge_pdfs_to_grid_2x2(files, titles, out_path, fmt=OUTPUT_FORMAT)

    print(f"\n🎉 全部完成！共生成 4 张 2×2 {OUTPUT_FORMAT.upper()}：")
    base_ext = f".{OUTPUT_FORMAT}"
    for _, _, out_path in tasks:
        real_path = os.path.splitext(out_path)[0] + base_ext
        print(f"   • {os.path.abspath(real_path)}")


if __name__ == "__main__":
    main()
