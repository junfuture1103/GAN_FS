import torch

import src
from torch import nn
from torch.nn import Module
from src.models import SNGANDModel
from src.models import WGANDModel
from src.models import WGANGPDModel
from src.models import JUNGANDModel
# imbalanced-learn 패키지
from imblearn.over_sampling import SMOTE
# device used for training
device: str = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())

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

class RVGAN(Base):
    def __init__(self):
        super().__init__(GANGModel(), GANDModel())

    def _fit(self, x):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=src.config_gan.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=src.config_gan.g_lr,
            betas=(0.5, 0.999),
        )

        # x = RoulettePositiveDataset().get_roulette_samples(len(PositiveDataset())).to(config.device)
        
        for _ in range(src.config_gan.epochs):
            for __ in range(src.config_gan.d_loops):
                self.d.zero_grad()

                prediction_real = self.d(x)
                loss_real = -torch.log(prediction_real.mean())
                
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = -torch.log(1 - prediction_fake.mean())
                loss = loss_real + loss_fake
                
                loss.backward()
                d_optimizer.step()

            for __ in range(src.config_gan.g_loops):
                self.g.zero_grad()
                real_x_hidden_output = self.d.hidden_output.detach()
                
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z)
                final_output = self.d(fake_x)
                fake_x_hidden_output = self.d.hidden_output
                real_x_hidden_distribution = normalize(real_x_hidden_output)
                fake_x_hidden_distribution = normalize(fake_x_hidden_output)
                hidden_loss = torch.norm(
                    real_x_hidden_distribution - fake_x_hidden_distribution,
                    p=2
                ) * src.config_gan.hl_lambda

                loss = -torch.log(final_output.mean()) + hidden_loss
                loss.backward()
                g_optimizer.step()


class WGAN(Base):
    def __init__(self):
        super().__init__(GANGModel(), WGANDModel())

    def _fit(self, x):
        d_optimizer = torch.optim.RMSprop(
            params=self.d.parameters(),
            lr=src.config_gan.d_lr
        )
        g_optimizer = torch.optim.RMSprop(
            params=self.g.parameters(),
            lr=src.config_gan.g_lr,
        )

        # x = PositiveDataset()[:][0].to(config.device)

        for epoch in range(src.config_gan.epochs):
            print(epoch,"/",src.config_gan.epochs," in WGAN")
            for __ in range(src.config_gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
                for p in self.d.parameters():
                    p.data.clamp_(*src.config_gan.wgan_clamp)
            for __ in range(src.config_gan.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()

           
class SNGAN(Base):
    def __init__(self):
        super().__init__(GANGModel(), SNGANDModel())

    def _fit(self, x):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=src.config_gan.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=src.config_gan.g_lr,
            betas=(0.5, 0.999),
        )

        # x = PositiveDataset()[:][0].to(device)

        for epoch in range(src.config_gan.epochs):
            print(epoch,"/",src.config_gan.epochs," in SNGAN")
            for __ in range(src.config_gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
            for __ in range(src.config_gan.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()     


class WGANGP(Base):
    def __init__(self):
        super().__init__(GANGModel(), WGANGPDModel())

    def _fit(self, x):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=src.config_gan.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=src.config_gan.g_lr,
            betas=(0.5, 0.999),
        )

        # x = PositiveDataset()[:][0].to(config.device)
        for epoch in range(src.config_gan.epochs):
            print(epoch,"/",src.config_gan.epochs," in WGANGP")
            for __ in range(src.config_gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                gradient_penalty = self._cal_gradient_penalty(x, fake_x)
                loss = loss_real + loss_fake + gradient_penalty
                loss.backward()
                d_optimizer.step()
            for __ in range(src.config_gan.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()

    def _cal_gradient_penalty(
            self,
            x: torch.Tensor,
            fake_x: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.rand(len(x), 1).to(device)
        interpolates = alpha * x + (1 - alpha) * fake_x
        interpolates.requires_grad = True
        disc_interpolates = self.d(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * src.config_gan.wgangp_lambda
        return gradient_penalty

class JUNGAN(Base):
    def __init__(self):
        super().__init__(GANGModel(), JUNGANDModel())

    def _fit(self, x):
        d_optimizer = torch.optim.RMSprop(
            params=self.d.parameters(),
            lr=src.config_gan.d_lr
        )
        g_optimizer = torch.optim.RMSprop(
            params=self.g.parameters(),
            lr=src.config_gan.g_lr,
        )

         # SMOTE 객체 생성
        smote = SMOTE(random_state=42)

        # x = PositiveDataset()[:][0].to(config.device) 
        for epoch in range(src.config_gan.epochs):
            print(epoch,"/",src.config_gan.epochs," in JUNGAN")
            for __ in range(src.config_gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
                for p in self.d.parameters():
                    p.data.clamp_(*src.config_gan.wgan_clamp)
            for __ in range(src.config_gan.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), src.models.z_size, device=device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()