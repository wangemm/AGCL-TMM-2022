from __future__ import print_function, division
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear


class AE_3views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_3views, self).__init__()
        dims0 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[0] * 0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)

        dims1 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)

        dims2 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[2] * 0.8)
            linshidim = int(linshidim)
            dims2.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims2.append(linshidim)

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)
        # encoder2
        self.enc2_1 = Linear(n_input[2], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        self.enc2_3 = Linear(dims2[1], dims2[2])
        self.z2_layer = Linear(dims2[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])
        # decoder2
        self.dec2_0 = Linear(n_z, n_z)
        self.dec2_1 = Linear(n_z, dims2[2])
        self.dec2_2 = Linear(dims2[2], dims2[1])
        self.dec2_3 = Linear(dims2[1], dims2[0])
        self.x2_bar_layer = Linear(dims2[0], n_input[2])

    def forward(self, x0, x1, x2):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3)
        # add directly
        z = (z0 + z1 + z2) / 3
        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x2_bar = self.x2_bar_layer(dec2_h3)

        return x0_bar, x1_bar, x2_bar, z, z0, z1, z2


class AE_2views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_2views, self).__init__()
        dims0 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[0] * 0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)

        dims1 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])

    def forward(self, x0, x1):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # add directly
        z = (z0 + z1) / 2
        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)

        return x0_bar, x1_bar, z, z0, z1
