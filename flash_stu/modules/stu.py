import torch
import torch.nn as nn
from flash_stu.utils.stu_utils import convolve, flash_convolve

try:
    from flashfftconv import FlashFFTConv
    flash_fft_available = True
except ImportError:
    flash_fft_available = False


class STU(nn.Module):
    def __init__(self, config, phi, n, feature_dim=None) -> None:
        super(STU, self).__init__()
        self.config = config
        self.register_buffer('phi', phi, persistent=False)
        self.n = n
        self.K = config.num_eigh
        self.d_in = feature_dim if feature_dim is not None else config.n_embd
        self.d_out = feature_dim if feature_dim is not None else config.n_embd
        self.use_hankel_L = config.use_hankel_L
        self.use_approx = config.use_approx
        self.flash_fft = (
            FlashFFTConv(self.n, dtype=torch.bfloat16)
            if config.use_flash_fft and flash_fft_available
            else None
        )
        if self.use_approx:
            self.M_inputs = nn.Parameter(
                torch.empty(self.d_in, self.d_out, dtype=config.torch_dtype)
            )
            self.M_filters = nn.Parameter(
                torch.empty(self.K, self.d_in, dtype=config.torch_dtype)
            )
        else:
            self.M_phi_plus = nn.Parameter(
                torch.empty(self.K, self.d_in, self.d_out, dtype=config.torch_dtype)
            )
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.empty(self.K, self.d_in, self.d_out, dtype=config.torch_dtype)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_tiling = (
            self.use_approx and 
            (self.config.d_in_tile is not None or self.config.d_out_tile is not None)
        )
        if use_tiling:
            return self._forward_tiled(x, self.config.d_in_tile, self.config.d_out_tile)
        else:
            return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        # Only compute both branches if we need them (standard Hankel)
        return_both = not self.use_hankel_L
        
        if self.use_approx:
            # Contract inputs and filters over the K and d_in dimensions, then convolve
            x_proj = x @ self.M_inputs
            phi_proj = self.phi @ self.M_filters
            if self.flash_fft:
                spectral_plus, spectral_minus = flash_convolve(
                    x_proj, phi_proj, self.flash_fft, self.use_approx, return_both
                )
            else:
                spectral_plus, spectral_minus = convolve(
                    x_proj, phi_proj, self.n, self.use_approx, return_both
                )
        else:
            # Convolve inputs and filters,
            if self.flash_fft:
                U_plus, U_minus = flash_convolve(
                    x, self.phi, self.flash_fft, self.use_approx, return_both
                )
            else:
                U_plus, U_minus = convolve(x, self.phi, self.n, self.use_approx, return_both)
            # Then, contract over the K and d_in dimensions
            spectral_plus = torch.tensordot(
                U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
            )
            if return_both:
                spectral_minus = torch.tensordot(
                    U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
                )

        return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus


    def _forward_tiled(
        self, 
        x: torch.Tensor, 
        d_in_tile: int = None, 
        d_out_tile: int = None
    ) -> torch.Tensor:
        """
        For nowm, I just implemented the diagonal block decomposition, when d_in == d_out.
        """
        if not self.use_approx:
            raise NotImplementedError("Tiling only supported for use_approx=True")
        
        if self.d_in != self.d_out:
            raise NotImplementedError(
                f"Tiling requires d_in == d_out for dimension matching in convolution. "
                f"Got d_in={self.d_in}, d_out={self.d_out}"
            )
        
        d_out_tile = d_out_tile or self.d_out
        d_in_tile = d_in_tile or self.d_in
        tile_size = min(d_in_tile, d_out_tile)
        
        if tile_size >= self.d_in:
            return self._forward_standard(x)
        
        return_both = not self.use_hankel_L
        return self._tile_diagonal_blocks(x, tile_size, return_both)
    
    def _compute_tile_conv(
        self,
        x_proj: torch.Tensor,
        phi_proj: torch.Tensor,
        return_both: bool
    ) -> torch.Tensor:
        if self.flash_fft:
            spectral_plus, spectral_minus = flash_convolve(
                x_proj, phi_proj, self.flash_fft, self.use_approx, return_both
            )
        else:
            spectral_plus, spectral_minus = convolve(
                x_proj, phi_proj, self.n, self.use_approx, return_both
            )
        
        return spectral_plus if not return_both else spectral_plus + spectral_minus
    
    def _tile_diagonal_blocks(
        self, 
        x: torch.Tensor, 
        tile_size: int, 
        return_both: bool
    ) -> torch.Tensor:
        results = []
        
        for i in range(0, self.d_in, tile_size):
            end = min(i + tile_size, self.d_in)
            
            x_tile = x[..., i:end]
            M_in_tile = self.M_inputs[i:end, i:end]
            M_filt_tile = self.M_filters[:, i:end]
            
            x_proj = x_tile @ M_in_tile
            phi_proj = self.phi @ M_filt_tile
            
            tile_result = self._compute_tile_conv(x_proj, phi_proj, return_both)
            results.append(tile_result)
        
        return torch.cat(results, dim=-1)