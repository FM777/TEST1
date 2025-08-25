import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

c = 3e8
freq = 3.5e9
lam = c / freq

def steering_vector(ny, nz, phi, theta, d=0.5*lam):
    m_y = np.arange(ny).reshape(-1, 1)
    m_z = np.arange(nz).reshape(1, -1)
    ky = np.sin(phi) * np.cos(theta)
    kz = np.sin(theta)
    a = np.exp(1j * 2 * np.pi * d * (m_y * ky + m_z * kz) / lam)
    return a.reshape(-1, 1)

def typeI_codebook(ny, nz, bits_phi=2, bits_theta=1):
    num_phi = 2 ** bits_phi
    num_theta = 2 ** bits_theta
    phi_list = np.linspace(-np.pi/2, np.pi/2, num_phi, endpoint=False)
    theta_list = np.linspace(-np.pi/4, np.pi/4, num_theta, endpoint=False)
    codebook = []
    for phi in phi_list:
        for theta in theta_list:
            w = steering_vector(ny, nz, phi, theta)
            w = w / np.sqrt(ny * nz)
            codebook.append(w)
    return codebook

def typeII_codebook(ny, nz, bits_phi=2, bits_theta=1, layers=2):
    base_codebook = typeI_codebook(ny, nz, bits_phi, bits_theta)
    codebook = []
    for i in range(len(base_codebook)):
        w1 = base_codebook[i]
        w2 = base_codebook[(i+1) % len(base_codebook)]
        w = np.hstack([w1, w2]) / np.sqrt(2)
        codebook.append(w)
    return codebook

def pmi_selection(h, codebook):
    best_idx = 0
    best_metric = -np.inf
    for idx, w in enumerate(codebook):
        metric = np.abs(np.conj(h.T) @ w)**2
        if metric > best_metric:
            best_metric = metric
            best_idx = idx
    return best_idx, codebook[best_idx], best_metric

def beam_pattern(w, ny, nz, d=0.5*lam, res=181, plane='horizontal'):
    angles = np.linspace(-np.pi/2, np.pi/2, res)
    gains = []
    for ang in angles:
        if plane == 'horizontal':
            phi, theta = ang, 0.0
        else:
            phi, theta = 0.0, ang
        a = steering_vector(ny, nz, phi, theta, d)
        af = np.conj(w.T) @ a
        gains.append(20*np.log10(np.abs(af))[0])
    return angles, np.array(gains)

def radiation_pattern_3d(w, ny, nz, d=0.5*lam, res=60):
    phi = np.linspace(-np.pi/2, np.pi/2, res)
    theta = np.linspace(-np.pi/2, np.pi/2, res)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    gains = np.zeros_like(phi_grid, dtype=float)
    for i in range(res):
        for j in range(res):
            a = steering_vector(ny, nz, phi_grid[i, j], theta_grid[i, j], d)
            af = np.conj(w.T) @ a
            gains[i, j] = 20*np.log10(np.abs(af))[0]
    return phi_grid, theta_grid, gains

def plot_2d(angles, gains, title='Beam Pattern'):
    plt.figure()
    plt.plot(np.degrees(angles), gains)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Gain (dB)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ','_')}.png")

def plot_3d(phi, theta, gains, title='3D Radiation Pattern'):
    X = gains * np.cos(theta) * np.cos(phi)
    Y = gains * np.cos(theta) * np.sin(phi)
    Z = gains * np.sin(theta)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ','_')}.png")

if __name__ == '__main__':
    ny, nz = 4, 4
    cb = typeI_codebook(ny, nz, bits_phi=3, bits_theta=2)
    user_phi = 0.2
    user_theta = -0.1
    h = steering_vector(ny, nz, user_phi, user_theta)
    idx, w_sel, metric = pmi_selection(h, cb)
    print(f'Selected PMI: {idx}, metric: {metric[0][0]:.2f}')
    angles, gains = beam_pattern(w_sel, ny, nz, plane='horizontal')
    plot_2d(angles, gains, title='Selected_Beam_Pattern')
