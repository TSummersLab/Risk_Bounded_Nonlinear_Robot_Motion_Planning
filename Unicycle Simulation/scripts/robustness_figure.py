
import math
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

theta = 35*(2*np.pi/360)
delta_theta_max = 25*(2*np.pi/360)
theta_min, theta_max = theta - delta_theta_max, theta + delta_theta_max
a1 = np.sin(delta_theta_max)
a2 = 1 - np.cos(delta_theta_max)
t_arc_buff = 0.01
t_arc = np.linspace(theta_min+t_arc_buff, theta_max-t_arc_buff, 1000)
t_cir = np.linspace(0, 2*np.pi, 1000)
x_arc, y_arc = np.cos(t_arc), np.sin(t_arc)
x_cir, y_cir = np.cos(t_cir), np.sin(t_cir)

v1 = np.array([-np.sin(theta), np.cos(theta)])
v2 = np.array([np.cos(theta), np.sin(theta)])


plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))

# Center point
plt.scatter(0, 0, color='k', zorder=100)

# Nominal point
plt.scatter(np.cos(theta), np.sin(theta), color='k', zorder=200)

# Connecting lines
bx, by = v1
cx, cy = v2
cx_min, cy_min = np.cos(theta_min), np.sin(theta_min)
cx_max, cy_max = np.cos(theta_max), np.sin(theta_max)

plt.plot([0, cx], [0, cy], lw=2, linestyle='--', color='k', alpha=0.5)
plt.plot([0, cx_min], [0, cy_min], lw=2, linestyle='--', color='k', alpha=0.5)
plt.plot([0, cx_max], [0, cy_max], lw=2, linestyle='--', color='k', alpha=0.5)
# plt.plot([cx_min, cx_max], [cy_min, cy_max], lw=2, linestyle='--', color='k', alpha=0.5)
plt.plot(np.array([cx-a2*cx, cx+a2*cx]), np.array([cy-a2*cy, cy+a2*cy]), lw=2, color='k')
plt.plot(np.array([cx_min, cx_max])+a2*cx, np.array([cy_min, cy_max])+a2*cy, lw=2, color='k')

plt.axvline(0, lw=2, alpha=1.0, color='k')
plt.axhline(0, lw=2, alpha=1.0, color='k')

# Bounding box
rect = Rectangle([cx_max, cy_max], width=2*a1, height=2*a2, angle=theta*360/(2*np.pi)-90, alpha=0.4, facecolor='tab:blue', edgecolor='none', zorder=500)
ax.add_patch(rect)


# Full circle
plt.plot(x_cir, y_cir, lw=2, linestyle='--', color='k', alpha=0.5)

# Circular arc segment
plt.plot(x_arc, y_arc, lw=6, zorder=400, color='tab:green')

# Direction arrows
plt.arrow(np.cos(theta), np.sin(theta), 0.2*np.cos(theta), 0.2*np.sin(theta),
          width=0.01, head_length=0.05, head_width=0.05, facecolor='k', edgecolor='k', zorder=300)

plt.arrow(np.cos(theta), np.sin(theta), -0.6*np.sin(theta), 0.6*np.cos(theta),
          width=0.01, head_length=0.05, head_width=0.05, facecolor='k', edgecolor='k', zorder=300)

plt.axis('equal')
# plt.axis('off')
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.2])
fig.savefig('robustness_figure.png', dpi=600)