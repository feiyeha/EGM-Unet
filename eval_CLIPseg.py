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
# def load_labels_from_mask(mask_path, file_names):
#     """
#     从掩码路径加载真实标签
    
#     参数:
#         mask_path: 掩码存放的根目录
#         file_names: 图像文件名列表 (如 ["Part02/0416", ...])
    
#     返回:
#         labels: 真实标签列表 [N, H, W] 值为0-3的整数
#     """
#     labels = []
#     for file in file_names:
#         mask_file = os.path.join(mask_path, file + ".png")
#         if os.path.exists(mask_file):
#             mask = Image.open(mask_file)
#             label = np.array(mask)  # 直接使用掩码值作为标签
#             # 确保标签值在0-3范围内
#             assert np.all((label >= 0) & (label <= 3)), f"发现非法标签值，应为0-3: {np.unique(label)}"
#             labels.append(label)
#         else:
#             raise FileNotFoundError(f"Mask file not found: {mask_file}")
#     return labels
# def search_best_alpha(clip_logits, unet_logits, labels, search_scale=[50, 50], search_step=[200, 20]):
#     """
#     通过不同alpha值循环测试，找到使准确率最高的最佳融合参数
    
#     参数:
#         clip_logits: CLIP模型输出的logits [N, C, H, W]
#         unet_logits: UNet模型输出的logits [N, C, H, W]
#         labels: 真实标签 [N, H, W] 值为0-3的整数
#         search_scale: alpha搜索范围 (min, max)
#         search_step: 搜索步数
    
#     返回:
#         最佳alpha值 (float)
#     """
#     # 生成alpha搜索列表
#     alpha_min, alpha_max = search_scale
#     alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    
    
#     best_alpha = 0.0  # 默认值
#     best_miou = 0.0
    
#     print("\n开始搜索最佳alpha值...")
#     print(f"测试的alpha列表: {alpha_list}")
#     print(f"标签类别: 0-背景, 1-车, 2-人行道, 3-盲道")
    
#     for alpha in alpha_list:
#         # 融合两个模型的预测（在softmax之前）
#         fused_logits = clip_logits + alpha * unet_logits
        
#         # 获取预测结果 (N, H, W)
#         preds = torch.argmax(fused_logits, dim=1).squeeze(0)
#         print(preds.size())
#         # 计算准确率
#         current_miou = calculate_miou(preds, labels)
        
#         print(f"alpha={alpha:.2f} \t mIoU={current_miou:.4f}")
        
#         # 更新最佳alpha
#         if current_miou > best_miou:
#             best_miou = current_miou
#             best_alpha = alpha
    
#     print(f"\n找到最佳alpha: {best_alpha:.2f} (mIoU={best_miou:.4f})")
#     return best_alpha

# class ConfusionMatrix:
#     """混淆矩阵类，用于计算mIoU"""
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.mat = None

#     def update(self, a, b):
#         n = self.num_classes
#         if self.mat is None:
#             self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
#         with torch.no_grad():
#             k = (a >= 0) & (a < n)
#             inds = n * a[k].to(torch.int64) + b[k]
#             self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

#     def compute(self):
#         h = self.mat.float()
#         # 处理分母为零的情况
#         denom = h.sum(1) + h.sum(0) - torch.diag(h)
#         # 避免除零错误
#         safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
#         iu = torch.diag(h) / safe_denom
#         # 将NaN值设为0
#         iu[torch.isnan(iu)] = 0.0
#         return iu.mean().item()
    
# def calculate_miou(pred, label, num_classes=4, device=None):
#     """
#     计算单张图像的mIoU
#     pred: torch.Tensor [H, W]
#     label: numpy数组 [H, W]
#     """
#     # 检查标签尺寸有效性
#     if label.size == 0 or label.shape[0] == 0 or label.shape[1] == 0:
#         print(f"警告: 标签尺寸无效 {label.shape}, 跳过mIoU计算")
#         return 0.0
    
#     # 转换预测图为numpy并确保为整数类型
#     pred = pred.cpu().numpy().astype(np.uint8)
    
#     # 调整预测图尺寸匹配标签
#     if pred.shape != label.shape:
#         try:
#             pred = cv2.resize(
#                 pred, 
#                 (label.shape[1], label.shape[0]),  # (width, height)
#                 interpolation=cv2.INTER_NEAREST
#             )
#         except Exception as e:
#             print(f"调整尺寸失败: pred-shape={pred.shape}, label-shape={label.shape}")
#             print(f"错误详情: {str(e)}")
#             return 0.0
    
#     # 转换为tensor
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     pred_tensor = torch.from_numpy(pred).to(device)
#     label_tensor = torch.from_numpy(label).to(device)
    
#     # 计算mIoU
#     confmat = ConfusionMatrix(num_classes)
#     confmat.update(label_tensor.flatten(), pred_tensor.flatten())
#     return confmat.compute()

# def main():
#     classes = 3  # exclude background
#     weights_path = "./save_weights/model_best.pth"
#     img_path = "./dataset/TP-Dataset/JPEGImages"
#     txt_path = "./dataset/TP-Dataset/Index/predict.txt"
#     save_result = "./predict/Part02"
#     mask_path = "./dataset/TP-Dataset/GroundTruth"
    
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
#     all_labels = load_labels_from_mask(mask_path, file_names)
#     total_time = 0
#     count = 0
    
#     for i,file in enumerate(file_names):
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
#         # best_alpha = 0.5  # 默认值
#         # 提供真实标签进行评估，评估的参数保存用于测试上
#         # 搜索最佳alpha
        
#         current_label = all_labels[i]
        
#         best_alpha = search_best_alpha(clip_logits, unet_logits, current_label)

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


# 上面是一次一次计算单张图片

# 这个是完成版，但是是四个类别的
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

# def load_labels_from_mask(mask_path, file_names):
#     """
#     从掩码路径加载真实标签
    
#     参数:
#         mask_path: 掩码存放的根目录
#         file_names: 图像文件名列表 (如 ["Part02/0416", ...])
    
#     返回:
#         labels: 真实标签列表 [N, H, W] 值为0-3的整数
#     """
#     labels = []
#     for file in file_names:
#         mask_file = os.path.join(mask_path, file + ".png")
#         if os.path.exists(mask_file):
#             mask = Image.open(mask_file)
#             label = np.array(mask)  # 直接使用掩码值作为标签
#             # 确保标签值在0-3范围内
#             assert np.all((label >= 0) & (label <= 3)), f"发现非法标签值，应为0-3: {np.unique(label)}"
#             labels.append(label)
#         else:
#             raise FileNotFoundError(f"Mask file not found: {mask_file}")
#     return labels

# def search_best_alpha(clip_logits_list, unet_logits_list, labels_list, search_scale=[0.1, 10.0], search_step=100):
#     """
#     在整个验证集上搜索最佳alpha值
    
#     参数:
#         clip_logits_list: CLIP模型输出的logits列表 [N, C, H, W]
#         unet_logits_list: UNet模型输出的logits列表 [N, C, H, W]
#         labels_list: 真实标签列表 [N, H, W] 值为0-3的整数
#         search_scale: alpha搜索范围 (min, max)
#         search_step: 搜索步数
    
#     返回:
#         最佳alpha值 (float)
#     """
#     alpha_min, alpha_max = search_scale
#     # alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
#     alpha_list = np.linspace(alpha_min, alpha_max, search_step)
    
#     best_alpha = 0.0
#     best_miou = 0.0
    
#     print("\n在整个验证集上搜索最佳alpha值...")
#     print(f"测试的alpha列表: 从{alpha_min}到{alpha_max}, 共{search_step}步")
    
#     num_classes = 4
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     for alpha in alpha_list:
#         # 创建全局混淆矩阵
#         confmat = ConfusionMatrix(num_classes)
        
#         for i in range(len(labels_list)):
#             # 融合两个模型的预测
#             fused_logits = clip_logits_list[i] + alpha * unet_logits_list[i]
#             pred = torch.argmax(fused_logits, dim=1).squeeze(0)
#             label = labels_list[i]
            
#             # 将预测转换为numpy数组
#             pred_np = pred.cpu().numpy().astype(np.uint8)
            
#             # 调整预测图尺寸匹配标签
#             if pred_np.shape != label.shape:
#                 try:
#                     pred_np = cv2.resize(
#                         pred_np, 
#                         (label.shape[1], label.shape[0]),  # (width, height)
#                         interpolation=cv2.INTER_NEAREST
#                     )
#                 except Exception as e:
#                     print(f"调整尺寸失败: pred-shape={pred_np.shape}, label-shape={label.shape}")
#                     print(f"错误详情: {str(e)}")
#                     continue
            
#             # 更新全局混淆矩阵
#             pred_tensor = torch.from_numpy(pred_np).to(device)
#             label_tensor = torch.from_numpy(label).to(device)
#             confmat.update(label_tensor.flatten(), pred_tensor.flatten())
        
#         # 计算整个验证集的mIoU
#         miou = confmat.compute()
#         print(f"alpha={alpha:.4f} \t mIoU={miou:.4f}")
        
#         # 更新最佳alpha
#         if miou > best_miou:
#             best_miou = miou
#             best_alpha = alpha
    
#     print(f"\n找到最佳alpha: {best_alpha:.4f} (mIoU={best_miou:.4f})")
#     return best_alpha

# class ConfusionMatrix:
#     """混淆矩阵类，用于计算mIoU"""
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.mat = None

#     def update(self, a, b):
#         n = self.num_classes
#         if self.mat is None:
#             self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
#         with torch.no_grad():
#             k = (a >= 0) & (a < n)
#             inds = n * a[k].to(torch.int64) + b[k]
#             self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

#     def compute(self):
#         h = self.mat.float()
#         # 处理分母为零的情况
#         denom = h.sum(1) + h.sum(0) - torch.diag(h)
#         # 避免除零错误
#         safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
#         iu = torch.diag(h) / safe_denom
#         # 将NaN值设为0
#         iu[torch.isnan(iu)] = 0.0
#         return iu.mean().item()
    
# def calculate_miou(pred, label, num_classes=4, device=None):
#     """
#     计算单张图像的mIoU
#     pred: torch.Tensor [H, W]
#     label: numpy数组 [H, W]
#     """
#     # 检查标签尺寸有效性
#     if label.size == 0 or label.shape[0] == 0 or label.shape[1] == 0:
#         print(f"警告: 标签尺寸无效 {label.shape}, 跳过mIoU计算")
#         return 0.0
    
#     # 转换预测图为numpy并确保为整数类型
#     pred = pred.cpu().numpy().astype(np.uint8)
    
#     # 调整预测图尺寸匹配标签
#     if pred.shape != label.shape:
#         try:
#             pred = cv2.resize(
#                 pred, 
#                 (label.shape[1], label.shape[0]),  # (width, height)
#                 interpolation=cv2.INTER_NEAREST
#             )
#         except Exception as e:
#             print(f"调整尺寸失败: pred-shape={pred.shape}, label-shape={label.shape}")
#             print(f"错误详情: {str(e)}")
#             return 0.0
    
#     # 转换为tensor
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     pred_tensor = torch.from_numpy(pred).to(device)
#     label_tensor = torch.from_numpy(label).to(device)
    
#     # 计算mIoU
#     confmat = ConfusionMatrix(num_classes)
#     confmat.update(label_tensor.flatten(), pred_tensor.flatten())
#     return confmat.compute()

# def save_alpha(alpha, file_path="best_alpha.txt"):
#     """保存最佳alpha值到文件"""
#     with open(file_path, 'w') as f:
#         f.write(str(alpha))
#     print(f"保存最佳alpha值到: {file_path}")

# def load_alpha(file_path="best_alpha.txt"):
#     """从文件加载最佳alpha值"""
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             return float(f.read().strip())
#     return 0.5  # 默认值

# def main():
#     classes = 3  # exclude background
#     weights_path = "./save_weights/model_best.pth"
#     img_path = "./dataset/TP-Dataset/JPEGImages"
#     txt_path = "./dataset/TP-Dataset/Index/predict.txt"
#     save_result = "./predict/Part02"
#     mask_path = "./dataset/TP-Dataset/GroundTruth"
#     alpha_file = "best_alpha.txt"
    
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
#         2: 100,    # sidewalk - dark gray
#         3: 255     # tactile paving - white
#     }
    
#     with open(os.path.join(txt_path), 'r') as f:
#         file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    
#     all_labels = load_labels_from_mask(mask_path, file_names)
#     total_time = 0
#     count = 0
    
#     # 存储所有logits用于全局alpha搜索
#     clip_logits_list = []
#     unet_logits_list = []
    
#     # 第一阶段: 收集所有logits
#     for i, file in enumerate(file_names):
#         original_img = Image.open(os.path.join(img_path, file + ".jpg"))
#         count += 1
        
#         # GRFB-UNet预测
#         img = data_transform(original_img).unsqueeze(0)
        
#         model.eval()
#         with torch.no_grad():
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
#         clip_logits = torch.nn.functional.interpolate(
#             clip_logits, size=unet_logits.shape[2:], 
#             mode='bilinear', align_corners=False
#         )
        
#         clip_logits_list.append(clip_logits)
#         unet_logits_list.append(unet_logits)
    
#     # 在整个验证集上搜索最佳alpha
#     best_alpha = search_best_alpha(clip_logits_list, unet_logits_list, all_labels)
    
#     # 保存最佳alpha值
#     save_alpha(best_alpha, alpha_file)
    
#     # 第二阶段: 使用最佳alpha生成预测结果
#     for i, file in enumerate(file_names):
#         original_img = Image.open(os.path.join(img_path, file + ".jpg"))
#         original_size = original_img.size
        
#         # 使用保存的logits
#         clip_logits = clip_logits_list[i]
#         unet_logits = unet_logits_list[i]
        
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
#         base_name = os.path.basename(file)
#         if not base_name.endswith(".png"):
#             base_name += ".png"
#         save_path = os.path.join(save_result, base_name)
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         mask.save(save_path)  # 修改了保存路径
#         print(f"保存预测结果到: {save_path}")  # 新增打印确认
    
#     fps = count / total_time if total_time > 0 else 0
#     print(f"Average FPS: {fps:.2f}")

# if __name__ == '__main__':
#     main()


# 这个就是二分类单盲道
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

def search_best_alpha(clip_logits_list, unet_logits_list, labels_list, search_scale=[0.1, 10.0], search_step=100):
    """
    在整个验证集上搜索最佳alpha值
    
    参数:
        clip_logits_list: CLIP模型输出的logits列表 [N, C, H, W]
        unet_logits_list: UNet模型输出的logits列表 [N, C, H, W]
        labels_list: 真实标签列表 [N, H, W] 值为0或1的整数
        search_scale: alpha搜索范围 (min, max)
        search_step: 搜索步数
    
    返回:
        最佳alpha值 (float)
    """
    alpha_min, alpha_max = search_scale
    alpha_list = np.linspace(alpha_min, alpha_max, search_step)
    
    best_alpha = 0.0
    best_miou = 0.0
    
    print("\n在整个验证集上搜索最佳alpha值...")
    print(f"测试的alpha列表: 从{alpha_min}到{alpha_max}, 共{search_step}步")
    
    num_classes = 2  # 二分类
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for alpha in alpha_list:
        # 创建全局混淆矩阵
        confmat = ConfusionMatrix(num_classes)
        
        for i in range(len(labels_list)):
            # 融合两个模型的预测
            fused_logits = clip_logits_list[i] + alpha * unet_logits_list[i]
            pred = torch.argmax(fused_logits, dim=1).squeeze(0)
            label = labels_list[i]
            
            # 将预测转换为numpy数组
            pred_np = pred.cpu().numpy().astype(np.uint8)
            
            # 调整预测图尺寸匹配标签
            if pred_np.shape != label.shape:
                try:
                    pred_np = cv2.resize(
                        pred_np, 
                        (label.shape[1], label.shape[0]),  # (width, height)
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception as e:
                    print(f"调整尺寸失败: pred-shape={pred_np.shape}, label-shape={label.shape}")
                    print(f"错误详情: {str(e)}")
                    continue
            
            # 更新全局混淆矩阵
            pred_tensor = torch.from_numpy(pred_np).to(device)
            label_tensor = torch.from_numpy(label).to(device)
            confmat.update(label_tensor.flatten(), pred_tensor.flatten())
        
        # 计算整个验证集的mIoU
        miou = confmat.compute()
        print(f"alpha={alpha:.4f} \t mIoU={miou:.4f}")
        
        # 更新最佳alpha
        if miou > best_miou:
            best_miou = miou
            best_alpha = alpha
    
    print(f"\n找到最佳alpha: {best_alpha:.4f} (mIoU={best_miou:.4f})")
    return best_alpha

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

def save_alpha(alpha, file_path="best_alpha.txt"):
    """保存最佳alpha值到文件"""
    with open(file_path, 'w') as f:
        f.write(str(alpha))
    print(f"保存最佳alpha值到: {file_path}")


def main():
    classes = 1  # 只有盲道一个前景类，加上背景共2类
    weights_path = "./save_weights/完整消融实验权重/扩充后的/A+B+C扩充后/model_最终版.pth"
    img_path = "./dataset/TP-Dataset/JPEGImages"
    txt_path = "./dataset/TP-Dataset/Index/val.txt"
    save_result = "./predict/A+B+C扩充后的+clipseg"
    mask_path = "./dataset/TP-Dataset/GroundTruth"
    alpha_file = "best_alpha.txt"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # CLIPSeg配置 - 只保留盲道和背景提示
    clip_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    clip_model.eval()
    clip_model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location='cpu'), strict=False)
    clip_model.to(device)
    prompts = ['background','Tactile paving']  # 只保留盲道和背景提示
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
    
    # 在整个验证集上搜索最佳alpha
    best_alpha = search_best_alpha(clip_logits_list, unet_logits_list, all_labels)
    
    # 保存最佳alpha值
    save_alpha(best_alpha, alpha_file)
    
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