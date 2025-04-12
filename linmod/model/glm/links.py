# Creating the base GLM structure with link function support

from abc import ABC, abstractmethod
import numpy as np


class LinkFunction(ABC):
    """Abstract base class for GLM link functions."""

    @abstractmethod
    def link(self, mu: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse(self, eta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, mu: np.ndarray) -> np.ndarray:
        """Derivative of the inverse link (used in IRLS)."""
        pass


class LogitLink(LinkFunction):
    """Logit link for logistic regression."""
    def link(self, mu: np.ndarray) -> np.ndarray:
        eps = 1e-8
        mu = np.clip(mu, eps, 1 - eps)
        return np.log(mu / (1 - mu))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-eta))

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)


class LogLink(LinkFunction):
    """Log link for Poisson regression."""
    def link(self, mu: np.ndarray) -> np.ndarray:
        return np.log(mu + 1e-8)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        return 1 / (mu + 1e-8)


class IdentityLink(LinkFunction):
    """Identity link (used in standard linear regression)."""
    def link(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def derivative(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

# These will plug into the GLM class later
link_functions = {
    "logit": LogitLink(),
    "log": LogLink(),
    "identity": IdentityLink()
}
