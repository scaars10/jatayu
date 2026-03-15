from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class ModelSelector(ABC):
    @abstractmethod
    def get_light_model(self) -> str:
        """Return the light model for easy tasks."""

    @abstractmethod
    def get_balanced_model(self) -> str:
        """Return the default model for everyday tasks."""

    @abstractmethod
    def get_large_model(self) -> str:
        """Return the stronger model for harder tasks."""


class StaticModelSelector(BaseModel, ModelSelector):
    model_config = ConfigDict(extra="forbid", frozen=True)

    balanced_model: str
    large_model: str
    light_model: str

    def get_balanced_model(self) -> str:
        return self.balanced_model

    def get_large_model(self) -> str:
        return self.large_model

    def get_light_model(self) -> str:
        return self.light_model
