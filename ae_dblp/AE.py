from torch import nn
from torch.nn import Linear
import torch

class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, n_input, n_z):
        super(AE_encoder, self).__init__()
        # For example, (batch_size, 1, 16, 16) for the provided input shape
        self.enc_1 = nn.Conv2d(n_input, n_input, kernel_size=3, stride=2, padding=1)

        self.enc_2 = nn.Conv2d(n_input, ae_n_enc_1, kernel_size=3, stride=1, padding=1)
        self.enc_3 = nn.Conv2d(ae_n_enc_1, ae_n_enc_1, kernel_size=3, stride=2, padding=1)

        self.enc_4 = nn.Conv2d(ae_n_enc_1, ae_n_enc_2, kernel_size=5, stride=3, padding=3)
        self.enc_5 = nn.Conv2d(ae_n_enc_2, 256, kernel_size=5, stride=3, padding=2)

        self.res_1 = nn.Conv2d(n_input, ae_n_enc_1, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(ae_n_enc_1)
        # The final layer will produce a 1D output
        self.z_layer1 = Linear(256 * 3 * 3, 2048)
        # self.z_layer2 = Linear(2048, 1024)
        # self.z_layer3 = Linear(1024, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))

        z_res = self.bn1(self.res_1(z))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z)+z_res)

        z = self.act(self.enc_4(z))
        z = self.act(self.enc_5(z))
        z = z.view(z.size(0), -1)
        # z = self.act(self.z_layer1(z))
        # z = self.act(self.z_layer2(z))
        z_ae = self.act(self.z_layer1(z))
        return z_ae


class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE_decoder, self).__init__()

        # self.z_layer1 = Linear(n_z, 1024)
        # self.z_layer2 = Linear(1024, 2048)
        self.z_layer1 = Linear(2048, 256 * 3 * 3)
        self.dec_1 = nn.ConvTranspose2d(256, ae_n_dec_1, kernel_size=5, stride=3, padding=2, output_padding=2)
        self.dec_2 = nn.ConvTranspose2d(ae_n_dec_1, ae_n_dec_2, kernel_size=5, stride=3, padding=3, output_padding=2)

        self.dec_3 = nn.ConvTranspose2d(ae_n_dec_2, ae_n_dec_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_4 = nn.ConvTranspose2d(ae_n_dec_2, n_input, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.dec_5 = nn.ConvTranspose2d(n_input, n_input, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.res_1 = nn.ConvTranspose2d(ae_n_dec_2, n_input, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn1 = nn.BatchNorm2d(n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.z_layer1(z_ae))
        # z = self.act(self.z_layer2(z))
        # z = self.act(self.z_layer3(z))
        z = z.view(z.size(0), -1, 3, 3)  # Adjust the dimensions to match the expected input for the first layer
        z = self.act(self.dec_1(z))
        z = self.act(self.dec_2(z))
        z_res = self.bn1(self.res_1(z))
        z = self.act(self.dec_3(z))
        z = self.act(self.dec_4(z)+z_res)
        x_hat = torch.sigmoid(self.dec_5(z))
        return x_hat


class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            n_input=n_input,
            n_z=n_z)

    def forward(self, x):
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae
