import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
server_bg = '#DDEBF7'
server_edge = '#2F5597'
client_bg = '#E2EFDA'
client_edge = '#548235'
inner_bg = 'white'

# 1. Base Station (Server)
ax.add_patch(patches.Rectangle((2.6, 4.9), 6.8, 2.8, edgecolor=server_edge, facecolor=server_bg, lw=2))
ax.text(6, 7.45, 'Base Station (Server)', ha='center', va='center', fontsize=22, fontweight='bold', fontfamily='sans-serif')

# Inner blocks for Server
ax.add_patch(patches.Rectangle((3.2, 6.15), 5.6, 0.75, edgecolor=server_edge, facecolor=inner_bg, lw=1))
ax.text(6, 6.53, r'Federated Aggregator (FedAvg / FedProx)' '\n' r'$W_{t+1} = \sum_{k} p_k W_{t}^k$', ha='center', va='center', fontsize=16, fontfamily='sans-serif')

ax.add_patch(patches.Rectangle((3.2, 5.2), 5.6, 0.75, edgecolor=server_edge, facecolor=inner_bg, lw=1))
ax.text(6, 5.58, 'Massive MIMO Antenna Array (8x8)', ha='center', va='center', fontsize=16, fontfamily='sans-serif')

# Draw some antennas on the BS
for i in range(5):
    ax.plot([3.8 + i*1.1, 3.8 + i*1.1], [5.2, 4.9], color=server_edge, lw=1.5)
    ax.plot([3.7 + i*1.1, 3.9 + i*1.1], [4.9, 4.9], color=server_edge, lw=1.5)
    ax.plot([3.6 + i*1.1, 3.8 + i*1.1], [5.0, 4.9], color=server_edge, lw=1.5)
    ax.plot([4.0 + i*1.1, 3.8 + i*1.1], [5.0, 4.9], color=server_edge, lw=1.5)

# 2. Clients
clients = [
    (0.5, 0.8, 'User Equipment 1', r'Non-IID Data $\mathcal{D}_1$'),
    (4.35, 0.8, 'User Equipment 2', r'Non-IID Data $\mathcal{D}_2$'),
    (8.2, 0.8, 'User Equipment K', r'Non-IID Data $\mathcal{D}_K$')
]

for x, y, title, data_label in clients:
    ax.add_patch(patches.Rectangle((x, y), 3.3, 3.0, edgecolor=client_edge, facecolor=client_bg, lw=2))
    ax.text(x+1.65, y+2.55, title, ha='center', va='center', fontsize=18, fontweight='bold', fontfamily='sans-serif')
    
    # Inner block for local training
    ax.add_patch(patches.Rectangle((x+0.2, y+1.65), 2.9, 0.75, edgecolor=client_edge, facecolor=inner_bg, lw=1))
    ax.text(x+1.65, y+2.02, r'Local Training (CNN/ResUNet)' '\n' r'$W_{t}^k \leftarrow \mathrm{Train}(W_t, \mathcal{D}_k)$', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
    
    # Inner block for data
    ax.add_patch(patches.Rectangle((x+0.2, y+0.45), 2.9, 0.75, edgecolor=client_edge, facecolor=inner_bg, lw=1))
    ax.text(x+1.65, y+0.82, data_label + '\n' + r'Noisy Channel $\tilde{H}$', ha='center', va='center', fontsize=14, fontfamily='sans-serif')

    # Draw client antenna
    ax.plot([x+1.65, x+1.65], [y+3.0, y+3.3], color=client_edge, lw=1.5)
    ax.plot([x+1.55, x+1.75], [y+3.3, y+3.3], color=client_edge, lw=1.5)
    ax.plot([x+1.45, x+1.65], [y+3.4, y+3.3], color=client_edge, lw=1.5)
    ax.plot([x+1.85, x+1.65], [y+3.4, y+3.3], color=client_edge, lw=1.5)

# Dots
ax.text(8.0, 2.2, '...', ha='center', va='center', fontsize=36, fontweight='bold')

# Wireless Channel Cloud
# ax.text(
#     6,
#     4.2,
#     'Wireless Channel (AWGN, Fading)',
#     ha='center',
#     va='center',
#     fontsize=9,
#     fontstyle='italic',
#     color='gray',
#     bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.15)
# )

# Arrows (slight curves reduce overlap with labels)
arrow_props_down = dict(
    arrowstyle="-|>",
    color="#2F5597",
    lw=2,
    connectionstyle="arc3,rad=0.06"
)
arrow_props_up = dict(
    arrowstyle="-|>",
    color="#548235",
    lw=2,
    ls='--',
    connectionstyle="arc3,rad=-0.06"
)

for i, (x, y, _, _) in enumerate(clients):
    ue_center_x = x + 1.65
    bs_center_x = 6.0 + (i - 1) * 2.2 # i=0: 3.8, i=1: 6.0, i=2: 8.2
    
    start_down = (bs_center_x - 0.15, 4.9)
    end_down = (ue_center_x - 0.15, 4.2)
    ax.annotate("", xy=end_down, xytext=start_down, arrowprops=arrow_props_down)
    
    start_up = (ue_center_x + 0.15, 4.2)
    end_up = (bs_center_x + 0.15, 4.9)
    ax.annotate("", xy=end_up, xytext=start_up, arrowprops=arrow_props_up)

# Label boxes keep text legible over lines
label_box = dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.2)

# Labels for UE1 (left)
ax.text(2.6, 4.55, r'Downlink:' '\n' r'Global Model $W_t$', ha='right', va='center', fontsize=14, color="#2F5597", fontweight='bold', fontfamily='sans-serif', bbox=label_box)
ax.text(3.2, 4.55, r'Uplink:' '\n' r'Local Update $W_t^k$', ha='left', va='center', fontsize=14, color="#548235", fontweight='bold', fontfamily='sans-serif', bbox=label_box)

# Labels for UEK (right)
ax.text(8.8, 4.55, r'Downlink:' '\n' r'Global Model $W_t$', ha='right', va='center', fontsize=14, color="#2F5597", fontweight='bold', fontfamily='sans-serif', bbox=label_box)
ax.text(9.4, 4.55, r'Uplink:' '\n' r'Local Update $W_t^K$', ha='left', va='center', fontsize=14, color="#548235", fontweight='bold', fontfamily='sans-serif', bbox=label_box)

plt.tight_layout()
plt.savefig('figures/system_architecture_diagram.png', dpi=300, bbox_inches='tight')
