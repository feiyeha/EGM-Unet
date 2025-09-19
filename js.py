# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # 选择第一个标签文件
# mask_path = "./datasets/0003.png"

# # 加载并显示图像
# mask = Image.open(mask_path)
# print(f"图像模式: {mask.mode}")  # 应该是 'L' (灰度) 或 'P' (调色板)
# print(f"图像大小: {mask.size}")

# # 显示图像
# plt.imshow(mask)
# plt.title("标签图像预览")
# plt.colorbar()
# plt.show()

# # 转换为numpy数组并检查值
# mask_array = np.array(mask)
# print(f"最小值: {np.min(mask_array)}")
# print(f"最大值: {np.max(mask_array)}")
# print(f"唯一值: {np.unique(mask_array)}")


import os
from PIL import Image
import numpy as np
import argparse
from collections import defaultdict

def analyze_pixel_values(folder_path):
    """
    分析文件夹中所有图像的像素值分布
    :param folder_path: 图像文件夹路径
    """
    # 支持的图像格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # 存储统计结果
    pixel_counter = defaultdict(int)
    file_pixel_values = {}
    problematic_files = []

    # 遍历文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                try:
                    # 打开图像并转换为灰度
                    with Image.open(file_path) as img:
                        img_gray = img.convert('L')
                        img_array = np.array(img_gray)
                        
                    # 统计像素值
                    unique_vals, counts = np.unique(img_array, return_counts=True)
                    file_values = set(unique_vals)
                    file_pixel_values[file_path] = file_values
                    
                    # 更新全局统计
                    for val, count in zip(unique_vals, counts):
                        pixel_counter[val] += count
                        
                except Exception as e:
                    problematic_files.append((file_path, str(e)))
    
    # 打印结果
    print("\n" + "="*50)
    print(f"分析完成! 共处理 {len(file_pixel_values)} 张图像")
    print("="*50)
    
    # 打印所有出现的像素值
    all_values = sorted(pixel_counter.keys())
    print("\n出现的所有像素值:", all_values)
    
    # 打印像素值频率统计
    print("\n像素值频率统计:")
    for val in sorted(pixel_counter.keys()):
        print(f"像素值 {val:3d}: 出现 {pixel_counter[val]:,} 次")
    
    # 打印包含255像素值的文件
    files_with_255 = [f for f, vals in file_pixel_values.items() if 255 in vals]
    if files_with_255:
        print("\n" + "!"*50)
        print(f"发现 {len(files_with_255)} 张图像包含像素值 255:")
        for file in files_with_255[:5]:  # 最多显示5个文件
            print(f"  - {file}")
        if len(files_with_255) > 5:
            print(f"  ...以及另外 {len(files_with_255)-5} 张图像")
        print("!"*50)
    
    # 打印有问题的文件
    if problematic_files:
        print("\n" + "#"*50)
        print(f"处理 {len(problematic_files)} 张图像时遇到问题:")
        for file, error in problematic_files:
            print(f"  - {file}: {error}")
        print("#"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析图像文件夹中的像素值分布')
    parser.add_argument('folder', type=str, help='要分析的图像文件夹路径')
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"错误: 路径 {args.folder} 不是有效的文件夹!")
        exit(1)
    
    print(f"开始分析文件夹: {args.folder}")
    analyze_pixel_values(args.folder)