import torch

import src
from torch import nn
from torch.nn import Module

# device used for training
device: str = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

class Base:
    def __init__(
            self,
            g: Module,
            d: Module,
    ):
        self.g = g.to(device)
        self.d = d.to(device)
        self.g.eval()
        self.d.eval()

    def fit(self, x):
        # self.logger.info('Started training')
        # self.logger.debug(f'Using device: {config.device}')
        self._fit(x)
        self.g.eval()
        self.d.eval()
        # self.logger.info(f'Finished training')

    def _fit(self, x):
        pass

    def generate_samples(self, z: torch.Tensor):
        return self.g(z)

class GANDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_1 = nn.Sequential(
            nn.Linear(src.models.x_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.2),
        )
        self.step_2 = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.hidden_output = None

        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        self.hidden_output = self.step_1(x)
        output = self.step_2(self.hidden_output)
        return output

class GANGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(src.models.z_size, 512, bias=False),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32, bias=False),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, src.models.x_size),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x_hat = self.process(x)
        return x_hat


class GAN(Base):
    def __init__(self):
        # Base(G, D)
        super().__init__(GANGModel(), GANDModel())

    def _fit(self, x):
        #discriminator
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=src.config_gan.d_lr,
            betas=(0.5, 0.999),
        )
        #generator
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=src.config_gan.g_lr,
            betas=(0.5, 0.999),
        )

        # x = src.datasets.PositiveDataset()[:][0].to(device)
        # print("x in GAN.fit : ", x)

        for _ in range(0, src.config_gan.epochs, -1):
            #d_train
            for __ in range(src.config_gan.d_loops):
                #역전파
                self.d.zero_grad()# 이번 step에서 쌓아놓은 파라미터들의 변화량을 0으로 초기화하여
                # 다음 step에서는 다음 step에서만의 변화량을 구하도록 함 
                
                prediction_real = self.d(x)
                loss_real = -torch.log(prediction_real.mean())
                
                #평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 랜덤 torch생성
                #랜덤 x값들 생성
                z = torch.randn(len(x), src.models.z_size, device=device)
                
                #detach는 복사
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                # get loss
                loss_fake = -torch.log(1 - prediction_fake.mean())
                loss = loss_real + loss_fake
                
                # 미분하여 손실함수에 끼친 변화량 구함
                # 파라미터들의 에러에 대한 변화도 계산 및 누적
                loss.backward()
                # optimizer에게 loss function를 효율적으로 최소화 할 수 있게 파라미터 수정 위탁
                # 옵티마이저를 이용해 손실함수 최적화하도록 파라미터 업데이트
                d_optimizer.step()

            #g_train
            for __ in range(src.config_gan.g_loops):
                #역전파
                self.g.zero_grad()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = -torch.log(prediction.mean())
                loss.backward()
                g_optimizer.step()
