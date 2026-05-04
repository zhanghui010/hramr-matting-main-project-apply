import os
import cv2
import numpy as np

def batch_matting_high_quality(original_dir, alpha_dir, output_dir):
    """
    高质量批量抠图函数：修复颜色失真与边缘模糊
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取并匹配文件名
    orig_files = {f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    alpha_files = {f for f in os.listdir(alpha_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    common_files = orig_files & alpha_files
    
    if not common_files:
        print("❌ 错误：两个文件夹中没有找到名称相同的图片文件！")
        return
        
    print(f"✅ 找到 {len(common_files)} 对匹配的图片，开始高质量抠图处理...\n")

    for filename in sorted(common_files):
        try:
            img_path = os.path.join(original_dir, filename)
            alpha_path = os.path.join(alpha_dir, filename)
            
            # 1. 读取原图（BGR格式）和 Alpha 遮罩（灰度图）
            original_img = cv2.imread(img_path)
            alpha_mask = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
            
            if original_img is None or alpha_mask is None:
                print(f"⚠️ 警告：无法读取文件 {filename}，已跳过。")
                continue

            # 2. 确保尺寸一致
            if original_img.shape[:2] != alpha_mask.shape[:2]:
                alpha_mask = cv2.resize(alpha_mask, (original_img.shape, original_img.shape))

            # 3. 【核心优化】提升计算精度与边缘平滑
            # 将图像和遮罩都转为 32位浮点数 (float32)，防止 0-255 整数运算时的精度丢失和溢出
            original_float = original_img.astype(np.float32)
            # 将 alpha 归一化到 0.0 - 1.0 之间
            alpha_float = alpha_mask.astype(np.float32) / 255.0
            
            # 对 Alpha 遮罩进行极其轻微的高斯模糊，消除锯齿，让发丝边缘过渡更自然
            # (3, 3) 的卷积核非常小，既能抗锯齿，又不会让边缘变糊
            alpha_float = cv2.GaussianBlur(alpha_float, (3, 3), 0)
            
            # 增加维度，使其能与 BGR 三通道直接相乘
            alpha_3channel = np.expand_dims(alpha_float, axis=2)

            # 4. 【核心优化】标准 Alpha Blending 公式
            # 融合公式：结果 = 前景 * Alpha + 背景 * (1 - Alpha)
            # 这里我们将“纯透明背景”作为背景，即背景色为 0。
            # 因此公式简化为：结果 = 前景 * Alpha + 0
            foreground = original_float * alpha_3channel
            
            # 5. 将计算好的前景转回 8位无符号整数 (uint8)
            foreground = foreground.astype(np.uint8)
            
            # 6. 合并为带透明通道的 BGRA 图像
            # 注意：这里的 alpha_mask 直接使用原始的 0-255 灰度图作为透明通道
            b, g, r = cv2.split(foreground)
            rgba = cv2.merge([b, g, r, alpha_mask])
            
            # 7. 保存为 PNG 格式
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, rgba)
            
            print(f"✨ 高质量抠图完成: {filename}")
            
        except Exception as e:
            print(f"❌ 处理 {filename} 时发生未知错误: {str(e)}")

# ================= 运行配置 =================
if __name__ == "__main__":
    # 请替换为你实际的文件夹路径
    ORIGINAL_FOLDER = r"C:\Users\65610\Desktop\HRAMR-Matting-main\dataset\Transparent-460\composited_images"  # 原始JPG文件夹
    ALPHA_FOLDER = r"C:\Users\65610\Desktop\HRAMR-Matting-main\HRAMR-Matting-MDRDLoss\Transparent-460\pred_alpha"        # Alpha通道JPG文件夹
    OUTPUT_FOLDER = r"C:\Users\65610\Desktop\HRAMR-Matting-main\output_results"    # 结果输出文件夹  
    
    batch_matting_high_quality(ORIGINAL_FOLDER, ALPHA_FOLDER, OUTPUT_FOLDER)