import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit, Int8ActPerTensorFloat
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import torch.fft


torch.manual_seed(0)
np.random.seed(0)


# We keep the quantizers outside of our models
class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
    bit_width = 6
    min_val = -2.0
    max_val = 2.0
    scaling_impl_type = ScalingImplType.CONST


class InputQuantizer1(Int8ActPerTensorFloat):
    bit_width = 6
    min_val = -1.0
    max_val = 1.0
    scaling_impl_type = ScalingImplType.CONST


class InputQuantizer_5bit(Int8ActPerTensorFloatMinMaxInit):
    bit_width = 5
    min_val = -2.0
    max_val = 2.0
    scaling_impl_type = ScalingImplType.CONST


class InputQuantizer1_5bit(Int8ActPerTensorFloat):
    bit_width = 5
    min_val = -1.0
    max_val = 1.0
    scaling_impl_type = ScalingImplType.CONST


class FullModelWithMoments(nn.Module):
    """
    This is the model we are using for the training puropses only, as some of the operations used are not
    supported by ONNX. In the evaluation and deployment stages, FinalModelWithMoments, defined below has to be used.
    We note, that this model contains two side-branches, i.e., self.classifier_momentum and self.classifier_conv, which
    are the side-classifiers used in the training phase only, and are discarded in the FinalModelWithMoments.
    """
    def __init__(self, a_bits=8, w_bits=8, filters_conv=64, filters_dense=128):
        super(FullModelWithMoments, self).__init__()
        torch.manual_seed(0)
        np.random.seed(0)

        self.momentum_features = nn.Sequential(qnn.QuantHardTanh(act_quant=InputQuantizer1),
                                               qnn.QuantLinear(32, 32, weight_bit_width=w_bits, bias=False),
                                               nn.BatchNorm1d(32),
                                               qnn.QuantReLU(bit_width=a_bits)
                                               )
        self.features = nn.Sequential(
            nn.AvgPool1d(2),
            qnn.QuantHardTanh(act_quant=InputQuantizer),

            qnn.QuantConv1d(2, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            nn.Flatten(),

            qnn.QuantLinear(filters_conv * 4, filters_dense, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits),
        )

        self.classifier = nn.Sequential(
            qnn.QuantLinear(filters_dense + 32, filters_dense, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),
            qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),)

        self.classifier_momentum = nn.Sequential(qnn.QuantLinear(32, 32, weight_bit_width=w_bits, bias=False),
                                                 nn.BatchNorm1d(32),
                                                 qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),
                                                 qnn.QuantLinear(32, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),)

        self.classifier_conv = nn.Sequential(qnn.QuantLinear(filters_dense, filters_dense, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),
            qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias))

        # A bunch of constant terms necessary to perform DFT and IDFT
        n = torch.arange(1024).float()
        self.kn = (n.unsqueeze(1).expand(1024, 1024) * n.unsqueeze(0).expand(1024, 1024)).unsqueeze(0).unsqueeze(0)
        h = torch.zeros(1024).float()
        h[0] = 1
        h[512] = 1
        h[1:512] = 2
        self.h = h.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def torch_dft(self, x):
        if x.is_cuda:
            real_dft = x.unsqueeze(-1) * torch.cos(2 * np.pi / x.shape[-1] * self.kn.cuda())
            real_dft = torch.sum(real_dft, dim=2, keepdim=True)
            imag_dft = - x.unsqueeze(-1) * torch.sin(2 * np.pi / x.shape[-1] * self.kn.cuda())
            imag_dft = torch.sum(imag_dft, dim=2, keepdim=True)
        else:
            real_dft = x.unsqueeze(-1) * torch.cos(2 * np.pi / x.shape[-1] * self.kn)
            real_dft = torch.sum(real_dft, dim=2, keepdim=True)
            imag_dft = - x.unsqueeze(-1) * torch.sin(2 * np.pi / x.shape[-1] * self.kn)
            imag_dft = torch.sum(imag_dft, dim=2, keepdim=True)
        return torch.cat([real_dft, imag_dft], dim=2)

    def torch_idft(self, x):
        if x.is_cuda:
            cos_part = - torch.cos(2 * np.pi / x.shape[-1] * self.kn.cuda())
            sin_part = torch.sin(2 * np.pi / x.shape[-1] * self.kn.cuda())
        else:
            cos_part = - torch.cos(2 * np.pi / x.shape[-1] * self.kn)
            sin_part = torch.sin(2 * np.pi / x.shape[-1] * self.kn)
        real_idft = - x[:, :, 0, :].unsqueeze(-1) * cos_part - x[:, :, 1, :].unsqueeze(-1) * sin_part
        imag_idft = x[:, :, 0, :].unsqueeze(-1) * sin_part - x[:, :, 1, :].unsqueeze(-1) * cos_part
        real_idft = torch.mean(real_idft, dim=2, keepdim=True)
        imag_idft = torch.mean(imag_idft, dim=2, keepdim=True)
        return torch.cat([real_idft, imag_idft], dim=2)

    @staticmethod
    def hilbert_torch(x):
        # Calculates Hilbert transform for signal analysis
        fft = torch.fft.fft(x)
        h = torch.zeros_like(x[0, 0, :])
        h[0] = h[x.shape[-1] // 2] = 1
        h[1:x.shape[-1] // 2] = 2
        return torch.view_as_real(torch.fft.ifft(fft * h.view(1, 1, -1), dim=-1)).transpose(2, 3)

    def torch_unwrap(self, x):
        # Port from np.unwrap
        dx = self.torch_diff(x)
        dx_m = ((dx + np.pi) % (2 * np.pi)) - np.pi
        dx_m[(dx_m == -np.pi) & (dx > 0)] = np.pi
        x_adj = dx_m - dx
        x_adj[dx.abs() < np.pi] = 0

        return x + x_adj.cumsum(-1)

    @staticmethod
    def torch_diff(x):
        # Port from np.diff
        return F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))

    @staticmethod
    def calculate_central_moments(x):
        # Returns central moments and uncentered mean of the signal
        central_moments = []
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_centered = x - x_mean
        central_moments.append(x_mean.squeeze(-1))
        moment = x_centered
        for i in range(3):
            moment = moment * x_centered
            central_moments.append(torch.mean(moment, dim=-1))
        return torch.cat(central_moments, dim=1)

    @staticmethod
    def complex_mul(x, y):
        # Simplified version of the function below
        real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
        imag = x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
        return torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1)

    @staticmethod
    def complex_abs(x):
        # Complex absolute value
        return torch.cat([(x[:, 0] ** 2 + x[:, 1] ** 2).unsqueeze(1), x[:, 0].unsqueeze(1) * 0.], dim=1)

    @staticmethod
    def complex_multiplication(x, y, conjugate_first=False, conjugate_second=False):
        # Performs complex-like multiplication of two tensors, if conjugate=True, multiplies x by y*
        if conjugate_first and conjugate_second:
            real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
            imag = - x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        elif conjugate_first:
            real = x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]
            imag = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        elif conjugate_second:
            real = x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]
            imag = - x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
        else:
            real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
            imag = x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
        return torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1)

    @staticmethod
    def save_complex_power(x, exponent):
        # return torch.view_as_real(torch.pow(torch.view_as_complex(x), exponent))
        return x

    @staticmethod
    def atan2my(x, y):
        return torch.sign(x) ** 2 * torch.atan(y / x) + (1 - torch.sign(x)) / 2 * (
                    1 + torch.sign(y) - torch.sign(y) ** 2) * np.pi

    def calculate_cumulants(self, x):
        # Higher order cumulants of the signal
        M20 = self.complex_multiplication(x, x)
        M21 = self.complex_multiplication(x, x, conjugate_second=True)
        M22 = self.complex_multiplication(x, x, conjugate_first=True, conjugate_second=True)
        M40 = self.complex_multiplication(M20, M20)
        M41 = self.complex_multiplication(M20, M21)
        M42 = self.complex_multiplication(M20, M22)
        M43 = self.complex_multiplication(M21, M22)
        M60 = self.complex_multiplication(M40, M20)
        M61 = self.complex_multiplication(M40, M21)
        M62 = self.complex_multiplication(M40, M22)
        M63 = self.complex_multiplication(M41, M22)

        M20, M21, M22, M40, M41, M42, M43, M60, M61, M62, M63 = torch.mean(M20, dim=-1), torch.mean(M21, dim=-1),\
                                                                torch.mean(M22, dim=-1), torch.mean(M40, dim=-1),\
                                                                torch.mean(M41, dim=-1), torch.mean(M42, dim=-1),\
                                                                torch.mean(M43, dim=-1), torch.mean(M60, dim=-1),\
                                                                torch.mean(M61, dim=-1),torch.mean(M62, dim=-1),\
                                                                torch.mean(M63, dim=-1)

        C20 = M20
        C21 = M21
        C40 = M40 - 3 * self.complex_mul(M20, M20)
        C41 = M41 - 3 * self.complex_mul(M20, M21)
        C42 = M42 - self.complex_abs(M20) ** 2 - 2 * self.complex_mul(M21, M21)
        C60 = M60 - 15 * self.complex_mul(M20, M40) + 30 * self.complex_mul(M20, self.complex_mul(M20, M20))
        C61 = M61 - 5 * self.complex_mul(M21, M40) - 10 * self.complex_mul(M20, M41) +\
              30 * self.complex_mul(M21, self.complex_mul(M20, M20))
        C62 = M62 - 6 * self.complex_mul(M20, M42) - 8 * self.complex_mul(M21, M41) - self.complex_mul(M22, M40) +\
              6 * self.complex_mul(M22, self.complex_mul(M20, M20)) +\
              24 * self.complex_mul(M20, self.complex_mul(M21, M21))
        C63 = M63 - 9 * self.complex_mul(M21, M42) + 12 * self.complex_mul(M21, self.complex_mul(M21, M21)) -\
              3 * self.complex_mul(M20, M43) - 3 * self.complex_mul(M22, M41) +\
              18 * self.complex_mul(M20, self.complex_mul(M21, M22))
        return torch.cat([C20, C21, self.save_complex_power(C40, 1./2.), self.save_complex_power(C41, 1./2.),
                             self.save_complex_power(C42, 1./2.), self.save_complex_power(C60, 1./3.),
                             self.save_complex_power(C61, 1./3.), self.save_complex_power(C62, 1./3.),
                             self.save_complex_power(C63, 1./3.)], dim=1)

    def calculate_statistics(self, x):
        # Calculate instantenous moments
        if self.training:
            analytic_signal = self.hilbert_torch(x)
        else:
            analytic_signal = self.hilbert_torch(x)
        instantaneous_amplitude = torch.sqrt(torch.sum(torch.pow(analytic_signal, 2), dim=2))
        instantaneous_amplitude = instantaneous_amplitude / torch.max(instantaneous_amplitude, dim=-1, keepdim=True)[0]

        instantaneous_phase = self.torch_unwrap(self.atan2my(analytic_signal[:, :, 0, :], analytic_signal[:, :, 1, :]))
        instantaneous_frequency = self.torch_diff(instantaneous_phase)
        instantaneous_frequency = instantaneous_frequency / torch.max(instantaneous_frequency, dim=-1, keepdim=True)[0]
        instantaneous_phase = instantaneous_phase / torch.max(instantaneous_phase, dim=-1, keepdim=True)[0]

        instantaneous_stats = torch.cat([torch.std(instantaneous_amplitude, dim=-1),
                                              torch.std(instantaneous_phase, dim=-1),
                                              torch.std(instantaneous_frequency, dim=-1)], dim=1)

        # Calculate moments
        moments = self.calculate_central_moments(x)

        # Calculate HOCs
        hocs = self.calculate_cumulants(x)

        # return torch.cat([moments, hocs], dim=1)
        return torch.cat([moments, hocs, instantaneous_stats], dim=1)

    def forward(self, x):
        statistics = self.calculate_statistics(x)
        x = self.features(x)
        statistics_features = self.momentum_features(statistics)
        logits_statistics = self.classifier_momentum(statistics_features)
        logits_signal = self.classifier_conv(x)
        x = torch.cat((x, statistics_features), dim=1)
        return self.classifier(x), logits_signal, logits_statistics


class FinalModelWithMoments(nn.Module):
    """
    This is the final model for evaluation purposes only. It requires much higher true computational resources
    than the FullModelWithMoments, as some of the efficient PyTorch-native operations were replaced by their equivalents,
    which are based on PyTorch primitive functions. It was necessary, because of the ONNX and finn compatibility.
    """
    def __init__(self, a_bits=8, w_bits=8, filters_conv=64, filters_dense=128):
        super(FinalModelWithMoments, self).__init__()
        torch.manual_seed(0)
        np.random.seed(0)

        self.momentum_features = nn.Sequential(qnn.QuantHardTanh(act_quant=InputQuantizer1_5bit),
                                               qnn.QuantLinear(32, 32, weight_bit_width=w_bits, bias=False),
                                               nn.BatchNorm1d(32),
                                               qnn.QuantReLU(bit_width=a_bits)
                                               )
        self.features = nn.Sequential(
            nn.AvgPool1d(2),
            qnn.QuantHardTanh(act_quant=InputQuantizer_5bit),

            qnn.QuantConv1d(2, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=a_bits),

            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=w_bits),

            # Increased quantization for this layer, as we observed the last conv layers to be more robust.
            qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, stride=2, weight_bit_width=4, bias=False),
            nn.BatchNorm1d(filters_conv),
            qnn.QuantReLU(bit_width=4),

            nn.Flatten(),

            qnn.QuantLinear(filters_conv * 4, filters_dense, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits),
        )

        self.classifier = nn.Sequential(
            qnn.QuantLinear(filters_dense + 32, filters_dense, weight_bit_width=w_bits, bias=False),
            nn.BatchNorm1d(filters_dense),
            qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),
            qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),)

        # A bunch of constants to avoid export errors in finn, which seems to dislike dynamic tensor creation on runtime
        n = torch.arange(1024).float()
        self.kn = (n.unsqueeze(1).expand(1024, 1024) * n.unsqueeze(0).expand(1024, 1024)).unsqueeze(0).unsqueeze(0)
        h = torch.zeros(1024).float()
        h[0] = 1
        h[512] = 1
        h[1:512] = 2
        self.h = h.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.triangular = torch.triu(torch.ones((1024, 1024))).float().unsqueeze(0).unsqueeze(0)

    def torch_dft(self, x):
        if x.is_cuda:
            real_dft = x.unsqueeze(-1) * torch.cos(2 * np.pi / x.shape[-1] * self.kn.cuda())
            real_dft = torch.sum(real_dft, dim=2, keepdim=True)
            imag_dft = - x.unsqueeze(-1) * torch.sin(2 * np.pi / x.shape[-1] * self.kn.cuda())
            imag_dft = torch.sum(imag_dft, dim=2, keepdim=True)
        else:
            real_dft = x.unsqueeze(-1) * torch.cos(2 * np.pi / x.shape[-1] * self.kn)
            real_dft = torch.sum(real_dft, dim=2, keepdim=True)
            imag_dft = - x.unsqueeze(-1) * torch.sin(2 * np.pi / x.shape[-1] * self.kn)
            imag_dft = torch.sum(imag_dft, dim=2, keepdim=True)
        return torch.cat([real_dft, imag_dft], dim=2)

    def torch_idft(self, x):
        if x.is_cuda:
            cos_part = - torch.cos(2 * np.pi / x.shape[-1] * self.kn.cuda())
            sin_part = torch.sin(2 * np.pi / x.shape[-1] * self.kn.cuda())
        else:
            cos_part = - torch.cos(2 * np.pi / x.shape[-1] * self.kn)
            sin_part = torch.sin(2 * np.pi / x.shape[-1] * self.kn)
        real_idft = - x[:, :, 0, :].unsqueeze(-1) * cos_part - x[:, :, 1, :].unsqueeze(-1) * sin_part
        imag_idft = x[:, :, 0, :].unsqueeze(-1) * sin_part - x[:, :, 1, :].unsqueeze(-1) * cos_part
        real_idft = torch.mean(real_idft, dim=2, keepdim=True)
        imag_idft = torch.mean(imag_idft, dim=2, keepdim=True)
        return torch.cat([real_idft, imag_idft], dim=2)

    def hilbert_torch(self, x):
        # Calculates Hilbert transform for signal analysis
        dft = self.torch_dft(x)
        if x.is_cuda:
            dft = dft * self.h.cuda()
        else:
            dft = dft * self.h
        return self.torch_idft(dft)

    def torch_unwrap(self, x):
        # Port from np.unwrap
        dx = self.torch_diff(x)
        dx_m = ((dx + np.pi) % (2 * np.pi)) - np.pi
        dx_m = torch.where((dx_m == -np.pi) & (dx > 0), dx_m * 0 + np.pi, dx_m)
        x_adj = dx_m - dx
        x_adj = torch.where(dx.abs() < np.pi, x_adj * 0, x_adj)
        if x.is_cuda:
            cumsum = torch.sum(self.triangular.cuda() * x_adj.unsqueeze(3), dim=2)
        else:
            cumsum = torch.sum(self.triangular * x_adj.unsqueeze(3), dim=2)
        return x + cumsum

    @staticmethod
    def torch_diff(x):
        # Port from np.diff
        return F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))

    @staticmethod
    def calculate_central_moments(x):
        # Returns central moments and uncentered mean of the signal
        central_moments = []
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_centered = x - x_mean
        central_moments.append(x_mean.squeeze(-1))
        moment = x_centered
        for i in range(3):
            moment = moment * x_centered
            central_moments.append(torch.mean(moment, dim=-1))
        return torch.cat(central_moments, dim=1)

    @staticmethod
    def complex_mul(x, y):
        # Simplified version of the function below
        real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
        imag = x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
        return torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1)

    @staticmethod
    def complex_abs(x):
        # Complex absolute value
        return torch.cat([(x[:, 0] ** 2 + x[:, 1] ** 2).unsqueeze(1), x[:, 0].unsqueeze(1) * 0.], dim=1)

    @staticmethod
    def complex_multiplication(x, y, conjugate_first=False, conjugate_second=False):
        # Performs complex-like multiplication of two tensors, if conjugate=True, multiplies x by y*
        if conjugate_first and conjugate_second:
            real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
            imag = - x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        elif conjugate_first:
            real = x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]
            imag = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        elif conjugate_second:
            real = x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]
            imag = - x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
        else:
            real = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
            imag = x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
        return torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1)

    @staticmethod
    def save_complex_power(x, exponent):
        # Decided to avoid calculating roots of complex numbers
        return x

    @staticmethod
    def atan2my(x, y):
        # Calculating atan2 without torch.atan2, which is unsupported by ONNX
        return torch.sign(x) ** 2 * torch.atan(y / x) + (1 - torch.sign(x)) / 2 * (
                    1 + torch.sign(y) - torch.sign(y) ** 2) * np.pi

    def calculate_cumulants(self, x):
        M20 = self.complex_multiplication(x, x)
        M21 = self.complex_multiplication(x, x, conjugate_second=True)
        M22 = self.complex_multiplication(x, x, conjugate_first=True, conjugate_second=True)
        M40 = self.complex_multiplication(M20, M20)
        M41 = self.complex_multiplication(M20, M21)
        M42 = self.complex_multiplication(M20, M22)
        M43 = self.complex_multiplication(M21, M22)
        M60 = self.complex_multiplication(M40, M20)
        M61 = self.complex_multiplication(M40, M21)
        M62 = self.complex_multiplication(M40, M22)
        M63 = self.complex_multiplication(M41, M22)

        M20, M21, M22, M40, M41, M42, M43, M60, M61, M62, M63 = torch.mean(M20, dim=-1), torch.mean(M21, dim=-1),\
                                                                torch.mean(M22, dim=-1), torch.mean(M40, dim=-1),\
                                                                torch.mean(M41, dim=-1), torch.mean(M42, dim=-1),\
                                                                torch.mean(M43, dim=-1), torch.mean(M60, dim=-1),\
                                                                torch.mean(M61, dim=-1),torch.mean(M62, dim=-1),\
                                                                torch.mean(M63, dim=-1)

        C20 = M20
        C21 = M21
        C40 = M40 - 3 * self.complex_mul(M20, M20)
        C41 = M41 - 3 * self.complex_mul(M20, M21)
        C42 = M42 - self.complex_abs(M20) ** 2 - 2 * self.complex_mul(M21, M21)
        C60 = M60 - 15 * self.complex_mul(M20, M40) + 30 * self.complex_mul(M20, self.complex_mul(M20, M20))
        C61 = M61 - 5 * self.complex_mul(M21, M40) - 10 * self.complex_mul(M20, M41) +\
              30 * self.complex_mul(M21, self.complex_mul(M20, M20))
        C62 = M62 - 6 * self.complex_mul(M20, M42) - 8 * self.complex_mul(M21, M41) - self.complex_mul(M22, M40) +\
              6 * self.complex_mul(M22, self.complex_mul(M20, M20)) +\
              24 * self.complex_mul(M20, self.complex_mul(M21, M21))
        C63 = M63 - 9 * self.complex_mul(M21, M42) + 12 * self.complex_mul(M21, self.complex_mul(M21, M21)) -\
              3 * self.complex_mul(M20, M43) - 3 * self.complex_mul(M22, M41) +\
              18 * self.complex_mul(M20, self.complex_mul(M21, M22))
        return torch.cat([C20, C21, self.save_complex_power(C40, 1./2.), self.save_complex_power(C41, 1./2.),
                             self.save_complex_power(C42, 1./2.), self.save_complex_power(C60, 1./3.),
                             self.save_complex_power(C61, 1./3.), self.save_complex_power(C62, 1./3.),
                             self.save_complex_power(C63, 1./3.)], dim=1)

    def calculate_statistics(self, x):
        # Calculate instantenous statistics
        analytic_signal = self.hilbert_torch(x)
        instantaneous_amplitude = torch.sqrt(torch.sum(torch.pow(analytic_signal, 2), dim=2))
        instantaneous_amplitude = instantaneous_amplitude / torch.max(instantaneous_amplitude, dim=-1, keepdim=True)[0]
        instantaneous_phase = self.torch_unwrap(self.atan2my(analytic_signal[:, :, 0, :], analytic_signal[:, :, 1, :]))
        instantaneous_frequency = self.torch_diff(instantaneous_phase)
        instantaneous_frequency = instantaneous_frequency / torch.max(instantaneous_frequency, dim=-1, keepdim=True)[0]
        instantaneous_phase = instantaneous_phase / torch.max(instantaneous_phase, dim=-1, keepdim=True)[0]
        instantaneous_stats = torch.cat([torch.std(instantaneous_amplitude, dim=-1),
                                              torch.std(instantaneous_phase, dim=-1),
                                              torch.std(instantaneous_frequency, dim=-1)], dim=1)

        # Calculate moments
        moments = self.calculate_central_moments(x)

        # Calculate HOCs
        hocs = self.calculate_cumulants(x)

        return torch.cat([moments, hocs, instantaneous_stats], dim=1)

    def forward(self, x):
        # This forward pass contains only the main branch of the network
        statistics = self.calculate_statistics(x)
        x = self.features(x)
        statistics_features = self.momentum_features(statistics)
        x = torch.cat((x, statistics_features), dim=1)
        return self.classifier(x)
