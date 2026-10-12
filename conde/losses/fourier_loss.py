import torch
import torch.nn as nn

class FourierLoss(nn.Module):
    """ Computes MSE loss directly on complex-valued Fourier coefficients. """

    def __init__(self):
        super(FourierLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        """
        img1, img2: tensors of shape (batch, channels, height, width)
        Returns: MSE loss on the full complex Fourier representation.
        """
        # Compute 2D FFT
        F1 = torch.fft.fft2(img1, norm="ortho")
        F2 = torch.fft.fft2(img2, norm="ortho")

        # Shift zero frequency to center
        F1_shifted = torch.fft.fftshift(F1)
        F2_shifted = torch.fft.fftshift(F2)

        loss_real = self.mse(F1_shifted.real, F2_shifted.real)
        loss_imag = self.mse(F1_shifted.imag, F2_shifted.imag)

        # Compute MSE directly on complex numbers (real & imaginary parts)
        return loss_real + loss_imag


class DoubleFourierLoss(nn.Module):
    """ Computes MSE loss on the double Fourier transform (FFT of FFT). """

    def __init__(self):
        super(DoubleFourierLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        """
        img1, img2: tensors of shape (batch, channels, height, width)
        Returns: MSE loss on the real and imaginary parts of the double Fourier transform.
        """
        # First Fourier Transform
        F1 = torch.fft.fft2(img1, norm="ortho")
        F2 = torch.fft.fft2(img2, norm="ortho")

        # Second Fourier Transform (FFT of the FFT)
        FF1 = torch.fft.fft2(F1, norm="ortho")
        FF2 = torch.fft.fft2(F2, norm="ortho")

        # Shift zero frequency to center
        FF1_shifted = torch.fft.fftshift(FF1)
        FF2_shifted = torch.fft.fftshift(FF2)

        # Compute MSE separately on real and imaginary parts
        loss_real = self.mse(FF1_shifted.real, FF2_shifted.real)
        loss_imag = self.mse(FF1_shifted.imag, FF2_shifted.imag)

        # Return total loss
        return loss_real + loss_imag

