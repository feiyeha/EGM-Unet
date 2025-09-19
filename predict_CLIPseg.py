# import os
# import time

# import torch
# import cv2
# from torchvision import transforms
# import numpy as np
# from PIL import Image

# from src import GRFBUNet


# def time_synchronized():
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     return time.time()


# def main():
#     classes = 3  # exclude background
#     weights_path = "./save_weights/model_best.pth"
#     img_path = "./datasets/TP-Dataset/JPEGImages"
#     txt_path = "./datasets/TP-Dataset/Index/predict.txt"
#     save_result = "./predict/Part02"

#     assert os.path.exists(weights_path), f"weights {weights_path} not found."
#     assert os.path.exists(img_path), f"image {img_path} not found."


#     mean = (0.709, 0.381, 0.224)
#     std = (0.127, 0.079, 0.043)

#     # get devices
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("using {} device.".format(device))

#     # create model
#     model = GRFBUNet(in_channels=3, num_classes=classes+1, base_c=32)

#     # load weights
#     model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
#     model.to(device)


#     total_time = 0
#     count = 0
#     with open(os.path.join(txt_path), 'r') as f:
#         file_name = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
#     for file in file_name:
#       original_img = Image.open(os.path.join(img_path, file + ".jpg"))
#       count = count +1
#       h = np.array(original_img).shape[0]
#       w = np.array(original_img).shape[1]



#       data_transform = transforms.Compose([transforms.Resize(565),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=mean, std=std)])
#       img = data_transform(original_img)
#       # expand batch dimension

#       img = torch.unsqueeze(img, dim=0)

#       model.eval()  # Entering Validation Mode
#       with torch.no_grad():
#         # init model
#         img_height, img_width = img.shape[-2:]
#         init_img = torch.zeros((1, 3, img_height, img_width), device=device)
#         model(init_img)

#         t_start = time_synchronized()
#         output = model(img.to(device))
#         print(output['out'])
#         t_end = time_synchronized()
#         total_time = total_time + (t_end - t_start)
#         print("inference+NMS time: {}".format(t_end - t_start))

#         prediction = output['out'].argmax(1).squeeze(0)
# #         这边获得argmax
#         prediction = prediction.to("cpu").numpy().astype(np.uint8)
#         prediction = cv2.resize(prediction, (w, h), interpolation = cv2.INTER_LINEAR)
#         # Change the pixel value corresponding to the foreground to 255 (white)
#         prediction[prediction == 1] = 255
#         # Set the pixels in the area of no interest to 0 (black)
#         prediction[prediction == 0] = 0
#         mask = Image.fromarray(prediction)
#         mask = mask.convert("L")
#         name = file[-4:]

#         if not os.path.exists(save_result):
#             os.makedirs(save_result)

#         mask.save(os.path.join(save_result, f'{name}.png'))
#     fps = 1 / (total_time / count)
#     print("FPS: {}".format(fps))

# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="pytorch GRGB-UNet predicting")
    
#     parser.add_argument("--weights_path", default="./save_weights/model_best.pth", help="The root of TP-Dataset ground truth list file")
#     parser.add_argument("--img_path", default="./data/TP-Dataset/JPEGImages", help="The path of testing sample images")
#     parser.add_argument("--txt_path", default="./data/TP-Dataset/Index/predict.txt", help="The path of testing sample list")
#     parser.add_argument("--save_result", default="./predict", help="The path of saved predicted results in images")

#     args = parser.parse_args()

#     return args

# if __name__ == '__main__':
#     args = parse_args()
#     main()
# import os
# import time
# import torch
# import cv2
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# from src import GRFBUNet
# from models.clipseg import CLIPDensePredT
# import glob

# def time_synchronized():
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     return time.time()

# def search_best_alpha(clip_logits, unet_logits, labels, alpha_list=[0.1, 0.5, 1.0, 2.0, 5.0]):
#     """
#     搜索最佳融合权重alpha
#     :param clip_logits: CLIPSeg的预测logits [N, C, H, W]
#     :param unet_logits: GRFB-UNet的预测logits [N, C, H, W]
#     :param labels: 真实标签 [H, W]
#     :param alpha_list: 搜索的alpha值列表
#     :return: 最佳alpha值
#     """
#     best_acc = 0
#     best_alpha = 0
    
#     # 将标签调整为与预测相同的尺寸
#     labels_resized = cv2.resize(labels.numpy(), (clip_logits.shape[2], clip_logits.shape[3]), 
#                               interpolation=cv2.INTER_NEAREST)
#     labels_resized = torch.from_numpy(labels_resized)
    
#     for alpha in alpha_list:
#         # 融合两个模型的预测
#         fused_logits = clip_logits + alpha * unet_logits
        
#         # 计算准确率
#         preds = torch.argmax(fused_logits, dim=1).squeeze(0)
#         correct = (preds == labels_resized).sum().item()
#         total = labels_resized.numel()
#         acc = correct / total
        
#         if acc > best_acc:
#             best_acc = acc
#             best_alpha = alpha
            
#     return best_alpha

# def main():
#     classes = 3  # exclude background
#     weights_path = "./save_weights/model_best.pth"
#     img_path = "./dataset/TP-Dataset/JPEGImages"
#     txt_path = "./dataset/TP-Dataset/Index/predict.txt"
#     save_result = "./predict/Part02"
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     # CLIPSeg配置
#     clip_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
#     clip_model.eval()
#     clip_model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location='cpu'), strict=False)
#     clip_model.to(device)
#     prompts = ['sidewalk', 'car', 'Tactile paving', 'background']
    
#     # 确保目录存在
#     os.makedirs(save_result, exist_ok=True)
    
#     # 加载GRFB-UNet模型
#     model = GRFBUNet(in_channels=3, num_classes=classes+1, base_c=32)
#     model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
#     model.to(device)
    
#     # 图像预处理
#     mean = (0.709, 0.381, 0.224)
#     std = (0.127, 0.079, 0.043)
#     data_transform = transforms.Compose([
#         transforms.Resize(565),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ])
    
#     # CLIPSeg预处理
#     clip_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         transforms.Resize((352, 352)),
#     ])
    
#     # 颜色映射
#     color_map = {
#         0: 0,      # background - black
#         1: 180,    # car - medium gray
#         2: 100,     # sidewalk - dark gray
#         3: 255      # tactile paving - white
#     }
    
#     with open(os.path.join(txt_path), 'r') as f:
#         file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    
#     total_time = 0
#     count = 0
    
#     for file in file_names:
#         # 加载原始图像
#         original_img = Image.open(os.path.join(img_path, file + ".jpg"))
#         original_size = original_img.size
#         count += 1
        
#         # GRFB-UNet预测
#         img = data_transform(original_img).unsqueeze(0)
        
#         model.eval()
#         with torch.no_grad():
#             # 初始化模型
#             img_height, img_width = img.shape[-2:]
#             init_img = torch.zeros((1, 3, img_height, img_width), device=device)
#             model(init_img)
            
#             t_start = time_synchronized()
#             output = model(img.to(device))
#             unet_logits = output['out']  # [1, 4, H, W]
#             t_end = time_synchronized()
#             total_time += (t_end - t_start)
#             print(f"GRFB-UNet inference time: {t_end - t_start:.4f}s")
        
#         # CLIPSeg预测
#         clip_img = clip_transform(original_img).unsqueeze(0)
#         with torch.no_grad():
#             t_start = time_synchronized()
#             clip_preds = clip_model(clip_img.repeat(len(prompts), 1, 1, 1), prompts)[0]  # [4, 1, 352, 352]
#             clip_logits = clip_preds.permute(1, 0, 2, 3)  # [1, 4, 352, 352]
#             t_end = time_synchronized()
#             print(f"CLIPSeg inference time: {t_end - t_start:.4f}s")
        
#         # 调整两个模型的输出尺寸一致
#         clip_logits = torch.nn.functional.interpolate(clip_logits, size=unet_logits.shape[2:], 
#                                                     mode='bilinear', align_corners=False)
        
#         # 搜索最佳alpha (这里假设你有真实标签，如果没有可以设置固定值)
#         # 注意: 实际预测时你可能没有真实标签，可以预先在验证集上确定最佳alpha
#         best_alpha = 0.5  # 默认值
#         # 如果你有真实标签，可以取消下面这行注释
#         # best_alpha = search_best_alpha(clip_logits, unet_logits, labels)
        
#         # 融合两个模型的预测
#         fused_logits = clip_logits + best_alpha * unet_logits
        
#         # 获取最终预测
#         prediction = torch.argmax(fused_logits, dim=1).squeeze(0)
#         prediction = prediction.cpu().numpy().astype(np.uint8)
        
#         # 调整到原始图像尺寸
#         prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_NEAREST)
        
#         # 应用颜色映射
#         color_prediction = np.zeros_like(prediction)
#         for class_id, color in color_map.items():
#             color_prediction[prediction == class_id] = color
        
#         # 保存结果
#         mask = Image.fromarray(color_prediction).convert("L")
#         name = file[-4:] if len(file) > 4 else file
#         mask.save(os.path.join(save_result, f'{name}.png'))
    
#     fps = count / total_time if total_time > 0 else 0
#     print(f"Average FPS: {fps:.2f}")

# if __name__ == '__main__':
#     main()


import os
import time
import torch
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
from src import GRFBUNet
from models.clipseg import CLIPDensePredT
import glob

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def load_labels_from_mask(mask_path, file_names):
    """
    从掩码路径加载真实标签
    
    参数:
        mask_path: 掩码存放的根目录
        file_names: 图像文件名列表 (如 ["Part02/0416", ...])
    
    返回:
        labels: 真实标签列表 [N, H, W] 值为0或1的整数 (0=背景, 1=盲道)
    """
    labels = []
    for file in file_names:
        mask_file = os.path.join(mask_path, file + ".png")
        if os.path.exists(mask_file):
            mask = Image.open(mask_file)
            label = np.array(mask)  # 直接使用掩码值作为标签
            
            # 将255转换为1，其他值保持为0
            binary_label = np.where(label == 255, 1, 0)
            
            # 确保标签值在0-1范围内
            assert np.all((binary_label >= 0) & (binary_label <= 1)), f"发现非法标签值，应为0-1: {np.unique(binary_label)}"
            labels.append(binary_label)
        else:
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
    return labels



class ConfusionMatrix:
    """混淆矩阵类，用于计算mIoU"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        # 处理分母为零的情况
        denom = h.sum(1) + h.sum(0) - torch.diag(h)
        # 避免除零错误
        safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        iu = torch.diag(h) / safe_denom
        # 将NaN值设为0
        iu[torch.isnan(iu)] = 0.0
        return iu.mean().item()
    
def calculate_miou(pred, label, num_classes=2, device=None):
    """
    计算单张图像的mIoU
    pred: torch.Tensor [H, W]
    label: numpy数组 [H, W]
    """
    # 检查标签尺寸有效性
    if label.size == 0 or label.shape[0] == 0 or label.shape[1] == 0:
        print(f"警告: 标签尺寸无效 {label.shape}, 跳过mIoU计算")
        return 0.0
    
    # 转换预测图为numpy并确保为整数类型
    pred = pred.cpu().numpy().astype(np.uint8)
    
    # 调整预测图尺寸匹配标签
    if pred.shape != label.shape:
        try:
            pred = cv2.resize(
                pred, 
                (label.shape[1], label.shape[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST
            )
        except Exception as e:
            print(f"调整尺寸失败: pred-shape={pred.shape}, label-shape={label.shape}")
            print(f"错误详情: {str(e)}")
            return 0.0
    
    # 转换为tensor
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_tensor = torch.from_numpy(pred).to(device)
    label_tensor = torch.from_numpy(label).to(device)
    
    # 计算mIoU
    confmat = ConfusionMatrix(num_classes)
    confmat.update(label_tensor.flatten(), pred_tensor.flatten())
    return confmat.compute()


def load_alpha(file_path="best_alpha.txt"):
    """从文件加载最佳alpha值"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return float(f.read().strip())
    return 0.5  # 默认值

def main():
    classes = 1  # 只有盲道一个前景类，加上背景共2类
    weights_path = "./save_weights/完整消融实验权重/扩充后的/A+B+C扩充后/model_最终版.pth"
    img_path = "./dataset/TP-Dataset/JPEGImages"
    txt_path = "./dataset/TP-Dataset/Index/predict.txt"
    save_result = "./predict/A+B+C扩充后的+clipseg+csa+long"
    mask_path = "./dataset/TP-Dataset/GroundTruth"
    alpha_file = "best_alpha.txt"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # CLIPSeg配置 - 只保留盲道和背景提示
    clip_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    clip_model.eval()
    clip_model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location='cpu'), strict=False)
    clip_model.to(device)
    prompts = ['Background','A textured pathway distinctly different from smooth pavement, with elevated linear elements and dot patterns that create a palpable surface variation, serving as a tactile map for blind navigation in public spaces.']
#     prompts = [
#     "Background: ordinary sidewalk pavement with smooth gray or brown concrete, no raised patterns, flat surface, may have small cracks or stains",
#     "Tactile paving for visually impaired people: hard rubber or concrete material with dense raised dots, each dot has a diameter of 5-8mm, center-to-center spacing of 15-20mm, surface rough to the touch; the color is bright yellow or orange, with obvious contrast to the surrounding gray sidewalk, forming a continuous strip 30-50cm wide along the edge of the sidewalk"
# ]
    # prompts = ['Background','Tactile paving for the visually impaired, with raised dots or raised strips arranged in a regular pattern']  # 只保留盲道和背景提示
    # prompts = ['Background','A specialized pathway for the blind, featuring a contrasting color compared to regular sidewalks and a textured surface with raised bumps or elongated strips.']
#     prompts = [
#     'Background without any tactile paving',
#     'Tactile paving for the visually impaired, with raised dots arranged in a regular pattern, usually found on sidewalks'
# ]

    # 确保目录存在
    os.makedirs(save_result, exist_ok=True)
    
    # 加载GRFB-UNet模型 - 修改为输出2类
    model = GRFBUNet(in_channels=3, num_classes=classes+1, base_c=32)  # classes+1=2
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    
    # 图像预处理
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([
        transforms.Resize(565),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # CLIPSeg预处理
    clip_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    
    # 颜色映射 - 修改为二分类
    color_map = {
        0: 0,      # background - black
        1: 255     # tactile paving - white
    }
    
    with open(os.path.join(txt_path), 'r') as f:
        file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    
    all_labels = load_labels_from_mask(mask_path, file_names)
    total_time = 0
    count = 0
    
    # 存储所有logits用于全局alpha搜索
    clip_logits_list = []
    unet_logits_list = []
    
    # 第一阶段: 收集所有logits
    for i, file in enumerate(file_names):
        original_img = Image.open(os.path.join(img_path, file + ".jpg"))
        count += 1
        
        # GRFB-UNet预测
        img = data_transform(original_img).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            
            t_start = time_synchronized()
            output = model(img.to(device))
            unet_logits = output['out']  # [1, 2, H, W] (二分类)
            t_end = time_synchronized()
            total_time += (t_end - t_start)
            print(f"GRFB-UNet inference time: {t_end - t_start:.4f}s")
        
        # CLIPSeg预测 - 现在只有两个提示，输出为[2, 1, 352, 352]
        clip_img = clip_transform(original_img).unsqueeze(0)
        with torch.no_grad():
            t_start = time_synchronized()
            clip_preds = clip_model(clip_img.repeat(len(prompts), 1, 1, 1), prompts)[0]  # [2, 1, 352, 352]
            clip_logits = clip_preds.permute(1, 0, 2, 3)  # [1, 2, 352, 352]
            t_end = time_synchronized()
            print(f"CLIPSeg inference time: {t_end - t_start:.4f}s")
        
        # 调整两个模型的输出尺寸一致
        clip_logits = torch.nn.functional.interpolate(
            clip_logits, size=unet_logits.shape[2:], 
            mode='bilinear', align_corners=False
        )
        
        clip_logits_list.append(clip_logits)
        unet_logits_list.append(unet_logits)
    
    # 加载验证集上的最佳alpha值
    best_alpha=load_alpha(alpha_file)
    
    # 第二阶段: 使用最佳alpha生成预测结果
    for i, file in enumerate(file_names):
        original_img = Image.open(os.path.join(img_path, file + ".jpg"))
        original_size = original_img.size
        
        # 使用保存的logits
        clip_logits = clip_logits_list[i]
        unet_logits = unet_logits_list[i]
        
        # 融合两个模型的预测
        fused_logits = clip_logits + best_alpha * unet_logits
        
        # 获取最终预测
        prediction = torch.argmax(fused_logits, dim=1).squeeze(0)
        prediction = prediction.cpu().numpy().astype(np.uint8)
        
        # 调整到原始图像尺寸
        prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_NEAREST)
        
        # 应用颜色映射
        color_prediction = np.zeros_like(prediction)
        for class_id, color in color_map.items():
            color_prediction[prediction == class_id] = color
        
        # 保存结果
        mask = Image.fromarray(color_prediction).convert("L")
        base_name = os.path.basename(file)
        if not base_name.endswith(".png"):
            base_name += ".png"
        save_path = os.path.join(save_result, base_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask.save(save_path)
        print(f"保存预测结果到: {save_path}")
    
    fps = count / total_time if total_time > 0 else 0
    print(f"Average FPS: {fps:.2f}")

if __name__ == '__main__':
    main()