import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Any

class foundation_fermi_Jastrow_MLP(nn.Module):
    n_layers: int
    d_model: int
    n_coups: int
    activation: Callable = nn.gelu
    initializer: Callable = nn.initializers.lecun_normal()
    is_disordered: bool = False
    out_activation: Callable | None = None
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        # 1. Separazione Spin e Couplings usando le fette (slicing) robuste
        x_spin = x[..., :-self.n_coups]
        coups = x[..., -self.n_coups:]
        
        # Calcoliamo il numero di siti (N) dinamicamente
        N_sites = x_spin.shape[-1] // 2
        
        # Rimodelliamo a (..., 2, N_sites) e sommiamo sull'asse degli spin (axis=-2)
        # keepdims=True mantiene la forma (..., 1, N_sites)
        x_charge = jnp.sum(
            x_spin.reshape(x_spin.shape[:-1] + (2, N_sites)), 
            axis=-2, 
            keepdims=True
        )

        if not self.is_disordered:
            # Coups ha forma (..., n_coups). Lo espandiamo a (..., n_coups, 1)
            expanded_coups = jnp.expand_dims(coups, axis=-1)
            # Lo ripetiamo per ogni sito: (..., n_coups, N_sites)
            expanded_coups = jnp.repeat(expanded_coups, N_sites, axis=-1)
        else:
            # Nel caso disordinato assumiamo che coups contenga un valore per sito.
            # Lo espandiamo a (..., 1, N_sites) per poterlo concatenare
            expanded_coups = jnp.expand_dims(coups, axis=-2)

        # Concateniamo la carica e i couplings lungo il penultimo asse
        # Risultato: (..., num_features, N_sites)
        x_combined = jnp.concatenate([x_charge, expanded_coups], axis=-2)

        # 4. Passaggio attraverso la MLP
        for i in range(self.n_layers):
            x_combined = nn.Dense(
                features=self.d_model, 
                kernel_init=self.initializer, 
                param_dtype=self.param_dtype,
                name=f"layer_{i}"
            )(x_combined)
            x_combined = self.activation(x_combined)
            
        # 5. Attivazione di output
        if self.out_activation is not None:
            x_combined = self.out_activation(x_combined)

        # 6. Somma finale
        # Sommiamo sia sulle features (axis=-2) che sulla dimensione d_model (axis=-1)
        # per ottenere lo scalare log-Jastrow totale
        out = jnp.sum(x_combined, axis=(-1, -2))

        return out