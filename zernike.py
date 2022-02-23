import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(img):
    img_ = np.array(img)
    Image.fromarray(img_).show()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def show_cercle(arr, x_shift, y_shift, R, cmap='Spectral', fig=None, part=(1, 1)):
    if not fig: fig, ax = plt.subplots()
    else: ax = fig.add_subplot(*part)
    im = plt.imshow(arr, cmap=cmap)
    patch = patches.Circle((x_shift, y_shift), radius=R-1, transform=ax.transData)
    im.set_clip_path(patch)
    ax.axis('off')
    if not fig: plt.show()


class Zernike:
    def __init__(self):
        self.img = None
        self.size = None
        self.R = None
        self.x_shift = None
        self.y_shift = None

    @staticmethod
    def load_img(img):
        if isinstance(img, str):
            img = np.array(Image.open(img).convert('L'))
        else:
            img = np.array(Image.fromarray(np.array(img)).convert('L'))
        return img / 255

    @staticmethod
    def load_coefs(filename):
        with open(filename, "r") as text_file:
            return eval(text_file.read())

    def crop_image(self, img, x, y, R, show=False):
        # print(img.shape)
        dx_left, dx_right, dy_up, dy_down = x, img.shape[1] - x, y, img.shape[0] - y
        if dx_left <= R:
            # print(dx_right)
            img = np.hstack(([[0 for k in range(R-dx_left)] for i in range(img.shape[0])], img))
        else:
            # print(dx_right)
            img = img[:, dx_left-R:]
        if dx_right <= R:
            # print(dx_left)
            img = np.hstack((img, [[0 for k in range(R-dx_right)] for i in range(img.shape[0])]))
        else:
            # print(dx_left)
            img = img[:, :-(dx_right-R)]
        if dy_up <= R:
            # print(dy_up)
            if R-dy_up > 0: img = np.vstack(([[0 for k in range(img.shape[1])] for i in range(R-dy_up)], img))
        else:
            # print(dy_up)
            img = img[dy_up-R:, :]
        if dy_down <= R:
            # print(dy_down)
            if R-dy_down > 0: img = np.vstack((img, [[0 for k in range(img.shape[1])] for i in range(R-dy_down)]))
        else:
            # print(dy_down)
            img = img[:-(dy_down-R), :]

        self.img = img
        self.size = img.shape[0]
        self.R = self.size / 2
        self.x_shift = self.y_shift = self.size // 2

        if show:
            show_cercle((img * 255).astype(np.uint8), self.x_shift, self.y_shift, self.R,
                        cmap='gray', fig=None, part=(1, 1))
        return img

    def R_mn(self, rho, m, n):
        sum = 0
        for k in range((n - m) // 2 + 1):
            # print(k)
            sum += (
                           (
                                   (-1) ** k *
                                   np.math.factorial(n - k)
                           ) / (
                                   np.math.factorial(k) *
                                   np.math.factorial((n + m) // 2 - k) *
                                   np.math.factorial((n - m) // 2 - k)
                           )
                   ) * rho ** (n - 2 * k)
        return sum

    def Z_mn(self, rho, phi, m, n, R):
        rho = rho.copy()
        phi = phi.copy()
        rho /= R
        if m < 0: return self.R_mn(rho, -m, n) * np.sin(-m * phi)
        return self.R_mn(rho, m, n) * np.cos(m * phi)

    def zernike_polynomials(self, img, max_depth, save=True):
        assert img.shape[0] == img.shape[1], "Изображение должно быть квадратным!"
        if max_depth < 1: return None

        X = np.zeros((self.size, self.size), dtype=int)
        if self.R % 2: x_ = np.arange(-self.y_shift, self.y_shift + 1)
        else: x_ = np.arange(-self.y_shift, self.y_shift)
        for i in range(self.size): X[i] = x_
        Y = np.transpose(X.copy()) * -1
        Z = np.sqrt(X ** 2 + Y ** 2)
        rho_arr = np.where(Z <= self.R, Z, 0)
        phi_arr = np.arctan2(Y, X)

        coefs = []
        print('Coeffs:')
        for depth in range(max_depth + 1):
            print(f'depth: {depth}')
            for m in range(-depth, depth + 1, 2):
                n = depth
                c_mn = (self.Z_mn(rho_arr, phi_arr, m, n, self.R) * img).sum() / (np.pi * (self.R ** 2))
                coefs += [((m, n), c_mn)]
        self.coefs = coefs
        if save:
            with open(f"coefs_{max_depth}.txt", "w") as text_file:
                text_file.write(f'{coefs}')
        return coefs

    def zernike_reconstruct_image(self, coefs, R, depth=None, show=True, save=False):

        size = 2 * R

        X = np.zeros((size, size), dtype=int)
        if R % 2: x_ = np.arange(-R, R + 1)
        else: x_ = np.arange(-R, R)
        for i in range(size): X[i] = x_
        Y = np.transpose(X.copy()) * -1
        Z = np.sqrt(X ** 2 + Y ** 2)
        rho_arr = np.where(Z <= R, Z, 0)
        phi_arr = np.arctan2(Y, X)

        n_max = max([n for ((m, n), c_mn) in coefs])
        assert n_max >= depth, "Depth should be <= max depth in coefficients"

        imaginary_image = np.sum([c_mn * self.Z_mn(rho_arr, phi_arr, m, n, R) for ((m, n), c_mn) in coefs
                                  if (m < 0) and (depth is not None and n <= depth)], axis=0)
        real_image = np.sum([c_mn * self.Z_mn(rho_arr, phi_arr, m, n, R) for ((m, n), c_mn) in coefs
                             if (m > 0) and (depth is not None and n <= depth)], axis=0)

        if show:
            fig = plt.figure(figsize=(15, 5))
            plt.subplot(1, 1, 1)
            plt.title(f'complex')
            plt.axis('off')
            show_cercle(imaginary_image, R, R, R, fig=fig, part=(1, 3, 1))
            plt.title('imaginary')
            show_cercle(real_image, R, R, R, fig=fig, part=(1, 3, 3))
            plt.title('real')
            show_cercle(real_image + imaginary_image, R, R, R, fig=fig, part=(1, 3, 2))
            if save is True:
                plt.savefig(f'FigDepth{depth}.png')
                plt.show()
        return real_image, imaginary_image, real_image + imaginary_image


if __name__ == "__main__":
    Z = Zernike()
    img = Z.load_img("star.jpg")
    print(img.shape)
    img = Z.crop_image(img=img, x=img.shape[1] // 2, y=img.shape[0] // 2, R=img.shape[0] // 2, show=True)
    Z.zernike_polynomials(img, max_depth=40)
    Z.zernike_reconstruct_image(Z.coefs, R=100, depth=40, save=True)
