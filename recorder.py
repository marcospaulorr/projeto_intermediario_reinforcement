"""
PygameRecord - Uma utilidade para gravar telas Pygame como GIFs.
Este módulo fornece uma classe, PygameRecord, que pode ser usada para gravar
animações Pygame e salvá-las como arquivos GIF.
"""

import pygame
from PIL import Image
import numpy as np


class PygameRecord:
    def __init__(self, filename: str, fps: int):
        self.fps = fps
        self.filename = filename
        self.frames = []

    def add_frame(self):
        curr_surface = pygame.display.get_surface()
        x3 = pygame.surfarray.array3d(curr_surface)
        x3 = np.moveaxis(x3, 0, 1)
        array = Image.fromarray(np.uint8(x3))
        self.frames.append(array)

    def save(self):
        if not self.frames:
            print("Nenhum frame para salvar!")
            return
            
        self.frames[0].save(
            self.filename,
            save_all=True,
            optimize=False,
            append_images=self.frames[1:],
            loop=0,
            duration=int(1000 / self.fps),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"Uma exceção do tipo {exc_type} ocorreu: {exc_value}")
        self.save()
        # Retorna False se você quer que exceções se propaguem, True para suprimi-las
        return False