import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

from deep_sort import DeepSort
from detector.YOLOv3.darknet import Darknet
from detector.YOLOv3.yolo_utils import do_detect
# 数据加载器

from detector.YOLOv3.darknet import Darknet
# from utils.utils import *
from utils.io import *
from utils.draw import *
from utils.parser import *

class MOTDataset(Dataset):
    def __init__(self, root_dir, seq_name, transform=None):
        self.root_dir = root_dir
        self.seq_name = seq_name
        self.transform = transform
        self.img_dir = os.path.join(root_dir, seq_name, 'img1')
        self.ann_file = os.path.join(root_dir, seq_name, 'gt', 'gt.txt')

        self.img_list = sorted(os.listdir(self.img_dir))
        self.annotations = pd.read_csv(self.ann_file, header=None,
                                       names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x',
                                              'y', 'z'])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        # image = Image.open(img_path).convert('RGB')
        #
        # if self.transform:
        #     image = self.transform(image)

        frame_id = idx + 1
        ann_frame = self.annotations[self.annotations['frame'] == frame_id]
        boxes = ann_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        ids = ann_frame['id'].values

        sample = {'image': img_path, 'boxes': boxes, 'ids': ids}
        return sample





def train_model(data_loader, model, deep_sort, device):
    model.to(device)
    # deep_sort.to(device)

    for i, sample in enumerate(data_loader):
        print("1")
        images = sample['image'].to(device)
        boxes = sample['boxes']
        ids = sample['ids']

        # 目标检测
        print("234")
        detections = do_detect(model, images, 0.5, 0.4, True)
        print("5")
        # DeepSORT跟踪
        outputs = deep_sort.update(detections, images)
        print("6")
        # 打印或保存结果
        print(f"Frame: {i + 1}, Outputs: {outputs}")

        if i == 10:  # 只运行前10个批次以示例
            break


# 示例使用
if __name__ == '__main__':
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor()
    ])

    # 实例化数据集
    root_dir = r"C:\Users\30505\Desktop\dataset\train"
    seq_name = 'MOT16-02'
    mot_dataset = MOTDataset(root_dir=root_dir, seq_name=seq_name, transform=transform)

    # 加载YOLOv3模型
    model = Darknet(r'C:\Users\30505\Desktop\deep_sort_pytorch-master\detector\YOLOv3\cfg\yolo_v3.cfg')
    model.load_weights(r'C:\Users\30505\Desktop\deep_sort_pytorch-master\detector\YOLOv3\weight\yolov3.weights')
    model.eval()
    print("123")
    data_loader = DataLoader(mot_dataset, batch_size=1, shuffle=False, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep_sort = DeepSort(r'/deep_sort/deep/checkpoint/resnet18-5c106cde.pth')
    train_model(data_loader, model, deep_sort, device)