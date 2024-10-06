import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard

parser = argparse.ArgumentParser(description = "Training Parameters")
parser.add_argument("--num_epochs", type = int, default = 100, help = "number of epochs for training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay of adam optimizer")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
args = parser.parse_args()

writer = SummaryWriter("./logs")  # 可以自定义日志路径

image_size = [args.channels, args.img_size, args.img_size]
device = "cuda" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        shape of z: [batch_size, latent_dim]
        """
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype = np.int32), 512),
            nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        """
        shape of image: [batch_size, args.channel, args.image_size, args.image_size]
        """
        prob = self.model(image.reshape(image.shape[0], -1))
        return prob


# dataset and dataloader
dataset = torchvision.datasets.MNIST(root = "./dataset",
                                     train = True,
                                     transform = torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Resize(args.img_size),
                                         ]
                                     ),
                                     download = True)
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

generator = Generator()
discriminator = Discriminator()

# optimizer of generator and discriminator
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = args.lr, betas = (args.b1, args.b2), weight_decay = args.weight_decay)
optimizer_G = torch.optim.Adam(generator.parameters(), lr = args.lr, betas = (args.b1, args.b2), weight_decay = args.weight_decay)

loss_fn = nn.BCELoss()

if device == "cuda":
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
else:
    print("use cpu for training")

for epoch in range(args.num_epochs):
    for i, mini_batch in enumerate(dataloader):
        gt_images, label = mini_batch
        batch_size = gt_images.shape[0]
        labels_one = torch.ones(batch_size, 1).to(device)
        labels_zero = torch.zeros(batch_size, 1).to(device)

        random_noise = torch.randn(args.batch_size, args.latent_dim)

        random_noise = random_noise.to(device)
        gt_images = gt_images.to(device)
        predicted_images = generator(random_noise)
        
        # train generator
        optimizer_G.zero_grad()
        recons_loss = torch.abs(predicted_images-gt_images).mean()
        recons_loss = 0.05 * recons_loss
        loss_g = recons_loss + loss_fn(discriminator(predicted_images), labels_one)
        loss_g.backward()
        optimizer_G.step()

        # train discriminator
        optimizer_D.zero_grad()
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        # NOTE: detach的作用是什么?
        # 在pytorch中, detach的作用是从计算图中分离张量, 使得该张量不参与梯度计算和反向传播。具体来说, detach()方法会返回一个与原张量相同数据的副本, 但该副本不会跟踪计算图。
        # 这里使用detach的作用是确保生成器的参数不会在训练判别器时被更新。否则，生成器在判别器的训练步骤中也会被错误的更新, 干扰生成器和判别器的独立优化过程。
        fake_loss = loss_fn(discriminator(predicted_images.detach()), labels_zero)
        loss_d = real_loss + fake_loss
        loss_d.backward()
        optimizer_D.step()

        if i % 100 == 0:
            print(f"step{len(dataloader) * epoch + i}, recon_loss:{recons_loss.item()}, generator_loss:{loss_g.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if i % 1000 == 0:
            image = predicted_images[:16].data
            torchvision.utils.save_image(image, f"./saved_images/image_{len(dataloader) * epoch + i}.png", nrow=4)
        
         # 将损失写入到 TensorBoard
        writer.add_scalar('Loss/Generator', loss_g.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch * len(dataloader) + i)

        torch.save(generator.state_dict(), f"./saved_model/generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"./saved_model/discriminator_epoch_{epoch}.pth")

writer.close()
