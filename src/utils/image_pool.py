"""Image history buffer for GAN discriminator stabilization.

Stores a history of generated images to reduce discriminator oscillation.
"""

import random

import torch


class ImagePool:
    """Buffer that holds previously generated images."""

    def __init__(self, pool_size: int = 50) -> None:
        """Initialize ImagePool.

        Args:
            pool_size: Maximum number of images to retain in buffer.
        """
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Query images from pool.

        With 50% probability returns an image from the pool, otherwise
        adds the incoming image to the pool and returns the incoming image.

        Args:
            images: Batch of newly generated images.

        Returns:
            Batch of images to train discriminator on.
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, 0)
