from __future__ import annotations
from .config import NIVEIS_RECURSOS
from .utils import put_text

def nivel_recursos(nivel: int) -> list[str]:
    return NIVEIS_RECURSOS.get(nivel, NIVEIS_RECURSOS[1])

def draw_rbac_overlay(frame, nivel: int):
    recursos = nivel_recursos(nivel)
    put_text(frame, f"Acesso concedido - Nivel {nivel}", (10, 30), scale=0.8, color=(0, 255, 255), thick=2)
    y = 60
    for r in recursos:
        put_text(frame, f"- {r}", (10, y), scale=0.6, color=(0, 255, 0), thick=2)
        y += 22
