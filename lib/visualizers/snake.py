from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
import cv2
from PIL import Image,ImageDraw
import shutil
mean = snake_config.mean
std = snake_config.std


class Visualizer:
    def visualize_ex(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(10, 5))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)

        plt.show()

    def visualize_training_box(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio

        ex_all = output['py'] # contour list
        print(len(ex_all), type(ex_all))
        for k in range(len(ex_all)):
            ex = ex_all[k] # if isinstance(ex, list) else ex
            if k > 0:
                ex_pre = ex_all[k-1]
                ex_pre = ex_pre.detach().cpu().numpy() * snake_config.down_ratio
            ex = ex.detach().cpu().numpy() * snake_config.down_ratio
            
            fig, ax = plt.subplots(1, figsize=(10, 5))
            fig.tight_layout()
            ax.axis('off')
            ax.imshow(inp)

            colors = np.array([
                [31, 119, 180],
                [255, 127, 14],
                [46, 160, 44],
                [214, 40, 39],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [126, 126, 126],
                [188, 189, 32],
                [26, 190, 207]
            ]) / 255.
            np.random.shuffle(colors)
            colors = cycle(colors)
            for i in range(len(ex)):
                color = next(colors).tolist()
                poly = ex[i]
                poly = np.append(poly, [poly[0]], axis=0)
 
                ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=4)

            plt.savefig('./demo_out/' + batch['name'], bbox_inches='tight', pad_inches=0)
        
    def visualize(self, output, batch):
        # self.visualize_ex(output, batch)
        self.visualize_training_box(output, batch)

    def visualize_contour(self, save_dir, output, batch):
        detection = output['detection']
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        
        if len(py) == 0:
            return
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]

        # draw contour and save
        image=Image.open(batch['meta']['path'][0])
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        for i in range(len(py)):
            draw = ImageDraw.Draw(image)
            tmp=[]
            for j in range(len(py[i])):
                tmp.append((py[i][j][0],py[i][j][1]))
            draw.polygon(tmp,fill=None,outline='red')

            image.save(save_dir+"/poly_test{}_{}.jpg".format(i,score[i]))

    def visualize_cmask(self, output):
        mask=output['mask']
        mask=mask.sigmoid().detach().cpu().numpy()
        thresh=0.4
        mask[mask>thresh]=255
        mask[mask<=thresh]=0
        cv2.imwrite("test.jpg",mask[0][0])

