# %%
import itertools
import multiprocessing
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Manager, Pool, Process

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

multiprocessing.set_start_method("spawn", True)

# %%
############################################################################################################
# |\nabla u| = f

# ((u - a1)^+)^2 + ((u - a2)^+)^2 + ((u - a3)^+)^2 = f^2 h^2


def calculate_unique_solution(a, b, f, h):

    d = abs(a - b)
    if d >= f * h:
        return min(a, b) + f * h
    else:
        return (a + b + np.sqrt(2 * f * f * h * h - (a - b) ** 2)) / 2


def sweeping_over_I_J_K(u, I, J, f, h):

    m = len(I)
    n = len(J)

    for (i, j) in itertools.product(I, J):
        if i == 0:
            uxmin = u[i + 1, j]
        elif i == m - 1:
            uxmin = u[i - 1, j]
        else:
            uxmin = np.min([u[i - 1, j], u[i + 1, j]])

        if j == 0:
            uymin = u[i, j + 1]
        elif j == n - 1:
            uymin = u[i, j - 1]
        else:
            uymin = np.min([u[i, j - 1], u[i, j + 1]])

        u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

        u[i, j] = np.min([u_new, u[i, j]])

    return u


def sweeping(u, v, h):

    f = 1.0 / v  ## slowness

    m, n = u.shape
    I = list(range(m))
    iI = I[::-1]
    J = list(range(n))
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, f, h)
    u = sweeping_over_I_J_K(u, iI, J, f, h)
    u = sweeping_over_I_J_K(u, iI, iJ, f, h)
    u = sweeping_over_I_J_K(u, I, iJ, f, h)

    return u


def eikonal_solve(u, f, h):

    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))
        print(f"Iteration {i}, Error = {err}")
        if err < 1e-6:
            break

    return u


############################################################################################################

# %% Fast for GPU
# def traveltime(event_loc, station_locs, time_table, rgrid, zgrid, h, sigma=1, bounds=None):

#     event_loc = event_loc.unsqueeze(0)
#     r = torch.sqrt(torch.sum((event_loc[:, :2] - station_locs[:, :2]) ** 2, dim=-1))
#     z = torch.abs(event_loc[:, 2] - station_locs[:, 2])

#     r = r.unsqueeze(-1).unsqueeze(-1)
#     z = z.unsqueeze(-1).unsqueeze(-1)

#     magn = (
#         1.0
#         / (2.0 * np.pi * sigma)
#         * torch.exp(-(((rgrid - r) / (np.sqrt(2 * sigma) * h)) ** 2 + ((zgrid - z) / (np.sqrt(2 * sigma) * h)) ** 2))
#     )
#     sum_magn = torch.sum(magn, dim=(-1, -2))
#     tt = torch.sum(time_table * magn, dim=(-1, -2)) / sum_magn

#     return tt

# %% Fast for CPU
def _interp(time_table, r, z, rgrid, zgrid, h):

    ir0 = (r - rgrid[0,0]).floor_divide(h).clamp(0, rgrid.shape[0] - 2).long()
    iz0 = (z - zgrid[0,0]).floor_divide(h).clamp(0, zgrid.shape[1] - 2).long()
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x1 = ir0 * h + rgrid[0,0]
    x2 = ir1 * h + rgrid[0,0]
    y1 = iz0 * h + zgrid[0,0]
    y2 = iz1 * h + zgrid[0,0]

    Q11 = time_table[ir0, iz0]
    Q12 = time_table[ir0, iz1]
    Q21 = time_table[ir1, iz0]
    Q22 = time_table[ir1, iz1]

    t = 1/(x2-x1)/(y2-y1) * (Q11*(x2-r)*(y2-z) + Q21*(r-x1)*(y2-z) + Q12*(x2-r)*(z-y1) + Q22*(r-x1)*(z-y1))

    return t


def traveltime(event_loc, station_locs, time_table, rgrid, zgrid, h, **kwargs):

    event_loc = event_loc.unsqueeze(0)
    r = torch.sqrt(torch.sum((event_loc[:, :2] - station_locs[:, :2]) ** 2, dim=-1))
    z = torch.abs(event_loc[:, 2]) - station_locs[:, 2]

    tt = _interp(time_table, r, z, rgrid, zgrid, h)

    return tt


# %%
def invert(time, loc, type, weight, up, us, rgrid, zgrid, h, t0=[0], loc0=[0, 0, 5], bounds=((-100,100),(-100,100),(0, 30)), device="cpu", add_eqt=False, gamma=0.1):

    # device = up.device
    loc0 = torch.tensor(loc0, dtype=torch.float32, requires_grad=True, device=device)
    t0 = torch.tensor(t0, dtype=torch.float32, requires_grad=True, device=device)
    bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
    p_index = torch.arange(len(type), device=device)[type == "P"]
    s_index = torch.arange(len(type), device=device)[type == "S"]
    obs_p = time[p_index]
    obs_s = time[s_index]
    loc_p = loc[p_index]
    loc_s = loc[s_index]
    weight_p = weight[p_index]
    weight_s = weight[s_index]

    # %% optimization
    optimizer = optim.LBFGS(params=[t0, loc0], max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loc0_ = torch.max(torch.min(loc0, bounds[:, 1]), bounds[:, 0])
        loc0_ = torch.nan_to_num(loc0_, nan=0)
        t0_ = t0
        if len(p_index) > 0:
            tt_p = traveltime(loc0_, loc_p, up, rgrid, zgrid, h, sigma=1)
            pred_p = t0_ + tt_p 
            loss_p = torch.mean(F.huber_loss(obs_p, pred_p, reduction="none") * weight_p)
            if add_eqt:
                dd_tt_p = (tt_p.unsqueeze(-1) - tt_p.unsqueeze(-2)) 
                dd_time_p = (obs_p.unsqueeze(-1) - obs_p.unsqueeze(-2))
                loss_p += gamma * torch.mean(F.huber_loss(dd_tt_p, dd_time_p, reduction="none") * weight_p.unsqueeze(-1) * weight_p.unsqueeze(-2))
            # loss_p = F.mse_loss(time_p, tt_p)
        else:
            loss_p = 0
        if len(s_index) > 0:
            tt_s = traveltime(loc0_, loc_s, us, rgrid, zgrid, h, sigma=1)
            pred_s = t0_ + tt_s
            loss_s = torch.mean(F.huber_loss(obs_s, pred_s, reduction="none") * weight_s)
            if add_eqt:
                dd_tt_s = (tt_s.unsqueeze(-1) - tt_s.unsqueeze(-2)) 
                dd_time_s = (obs_s.unsqueeze(-1) - obs_s.unsqueeze(-2))
                loss_s += gamma * torch.mean(F.huber_loss(dd_tt_s, dd_time_s, reduction="none") * weight_s.unsqueeze(-1) * weight_s.unsqueeze(-2))
            # loss_s = F.mse_loss(time_s, tt_s)
        else:
            loss_s = 0
        loss = loss_p + loss_s
        # loss.backward(retain_graph=True)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure().item()

    return t0.detach().cpu(), loc0.detach().cpu(), loss


# %%
def run(catalog, event_index, picks, center, shift_z, up, us, rgrid, zgrid, h, bounds=((-100,100),(-100,100),(0, 30)), device="cpu", add_eqt=False, gamma=0.1):

    if (event_index == -1):
        return

    picks_ = picks

    # %%
    picks_tmin = picks_["timestamp"].min()
    picks_tt = (picks_["timestamp"] - picks_tmin).dt.total_seconds()
    picks_time = torch.tensor(picks_tt.values, dtype=torch.float32).to(device)
    picks_loc = torch.tensor(picks_[["x_km", "y_km", "z_km"]].values, dtype=torch.float32).to(device)
    picks_type = picks_["type"].values
    picks_weight = torch.tensor(picks_["prob"].values, dtype=torch.float32).to(device)

    # %%
    up = up.to(device)
    us = us.to(device)
    rgrid = rgrid.to(device)
    zgrid = zgrid.to(device)

    # %%
    event_t0, event_loc0, loss = invert(
        picks_time, picks_loc, picks_type, picks_weight, up, us, rgrid, zgrid, h, device=device, add_eqt=add_eqt, gamma=gamma, bounds=bounds
    )
    origin_time = picks_tmin + pd.to_timedelta(event_t0.item(), unit="s")
    longitude = event_loc0[0].item() / 111.2 / np.cos(np.deg2rad(center[1])) + center[0]
    latitude = event_loc0[1].item() / 111.2 + center[1]
    depth = np.abs(event_loc0[2].item()) - shift_z

    catalog.append(
        {
            "event_index": event_index,
            "time": origin_time,
            "latitude": latitude,
            "longitude": longitude,
            "depth_km": depth,
            "error": loss,
        }
    )
    if len(catalog) % 100 == 0:
        print(f"{datetime.now()}: {len(catalog)} events inverted.")


# %%
if __name__ == "__main__":

    # %%
    @dataclass
    class ArgParser:
        # device: str = "cuda"
        device: str = "cpu"
        add_eqt: bool = False
        gamma: float = 1.0
    args = ArgParser()

    # %% Set domain
    min_longitude, max_longitude, min_latitude, max_latitude = [-118.1, -117.1, 35.4, 36.3]
    center = [(min_longitude + max_longitude) / 2, (min_latitude + max_latitude) / 2]
    deg2km = 111.2
    xlim = [(min_longitude - center[0])*deg2km*np.cos(np.deg2rad(center[1])), (max_longitude - center[0])*deg2km*np.cos(np.deg2rad(center[1]))]
    ylim = [(min_latitude - center[1])*deg2km, (max_latitude - center[1])*111.2]
    zlim = [0, 50]
    bounds = (xlim, ylim, zlim)
    h = 1.0
    edge_grids = 3

    rlim = [0, ((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2) ** 0.5]

    rgrid = np.arange(rlim[0]-edge_grids*h, rlim[1], h)
    zgrid = np.arange(zlim[0]-edge_grids*h, zlim[1], h)
    m = len(rgrid)
    n = len(zgrid)
    

    # %% Calculate traveltime table

    ## California
    zz = [0.0, 5.5, 16.0, 32.0, zlim[1]]
    vp = [5.5, 5.5,  6.7,  7.8,   7.8]
    vp_vs_ratio = 1.73

    ## Turkey
    # zz = [0.0, 5.0, 20.0, 30.0, 32.0, 38.0, 43.5, zlim[1]]
    # vp = [5.95, 6.00, 6.29, 6.47, 6.74, 7.67, 7.82, 7.82]
    # vp_vs_ratio = 1.76

    vp1d = np.interp(zgrid, zz, vp)
    vs1d = vp1d / vp_vs_ratio
    plt.figure()
    plt.plot(zz, vp, "o")
    plt.plot(zgrid, vp1d)
    plt.plot(zgrid, vs1d)
    plt.savefig("velocity_model.png")

    # %%
    vp = np.ones((m, n)) * vp1d
    vs = np.ones((m, n)) * vs1d

    up = 1000 * np.ones((m, n))
    up[edge_grids, edge_grids] = 0.0
    up = eikonal_solve(up, vp, h)

    us = 1000 * np.ones((m, n))
    us[edge_grids, edge_grids] = 0.0
    us = eikonal_solve(us, vs, h)

    up = torch.tensor(up, dtype=torch.float32)
    us = torch.tensor(us, dtype=torch.float32)
    rgrid = torch.tensor(rgrid, dtype=torch.float32)
    zgrid = torch.tensor(zgrid, dtype=torch.float32)
    rgrid, zgrid = torch.meshgrid(rgrid, zgrid, indexing="ij")

    # %% Read picks and stations
    # os.system("curl -O -J -L https://osf.io/945dq/download")
    # os.system("curl -O -J -L https://osf.io/gwxtn/download")
    # os.system("curl -O -J -L https://osf.io/km97w/download")

    picks = pd.read_csv("picks_gamma.csv", sep="\t", parse_dates=["timestamp"])
    picks["event_index"] = picks["event_idx"]
    picks["type"] = picks["type"].apply(lambda x: x.upper())
    stations = pd.read_csv("stations.csv", sep="\t")
    stations["id"] = stations["station"]
    picks = picks.merge(stations, on="id")

    # %%
    # picks = pd.read_csv("gamma_picks.csv", parse_dates=["phase_time"])
    # picks["id"] = picks["station_id"]
    # picks["timestamp"] = picks["phase_time"]
    # picks["amp"] = picks["phase_amp"]
    # picks["type"] = picks["phase_type"].apply(lambda x: x.upper())
    # picks["prob"] = picks["phase_score"]
    # stations = pd.read_json("stations.json", orient="index")
    # stations["id"] = stations.index
    # stations["elevation(m)"] = stations["elevation_m"]
    # picks = picks.merge(stations, on="id")

    # %% Prepare input data
    # center = [stations["longitude"].mean(), stations["latitude"].mean(), 0]
    shift_z = picks["elevation(m)"].max() / 1000
    picks["x_km"] = (picks["longitude"] - center[0]) * deg2km * np.cos(np.deg2rad(center[1]))
    picks["y_km"] = (picks["latitude"] - center[1]) * deg2km
    picks["z_km"] = -picks["elevation(m)"] / 1000 + shift_z

    # %% parallel inversion
    mangaer = mp.Manager()
    catalog = mangaer.list()
    num_cpu = mp.cpu_count()
    # num_cpu = 1
    num_gpu = torch.cuda.device_count()
    print("num_cpu", num_cpu)
    print(f"Total number of events: {len(picks['event_index'].unique())}")

    with mp.Pool(num_cpu) as pool:
        pool.starmap(
            run,
            [
                (
                    catalog,
                    event_index,
                    picks[picks["event_index"] == event_index],
                    center,
                    shift_z,
                    up,
                    us,
                    rgrid,
                    zgrid,
                    h,
                    bounds,
                    args.device + f":{i%num_gpu}",
                    args.add_eqt,
                    args.gamma,
                )
                for i, event_index in enumerate(list(picks["event_index"].unique()))
                # for i, event_index in enumerate(list(picks["event_index"].unique())[:2000])
            ],
        )

    catalog = pd.DataFrame(list(catalog))
    catalog.to_csv("catalog.csv", index=False)

    # %%
    catalog = pd.read_csv("catalog.csv", parse_dates=["time"])
    # catalog_gamma = pd.read_csv("catalog_gamma.csv", sep="\t", parse_dates=["time"])
    # catalog_gamma["depth_km"] = catalog_gamma["depth(m)"] / 1000

    # %%
    fig, axis = plt.subplots(3, 1, squeeze=False)
    axis[0, 0].hist(catalog["error"], bins=100, range=(0, 5))
    axis[1, 0].hist(catalog["depth_km"], bins=100, range=zlim)
    fig.savefig("error.png", dpi=300, bbox_inches="tight")

    # %%
    catalog = catalog[catalog["error"] < 5.0]
    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(5, 8), gridspec_kw={"height_ratios": [4, 1, 1]})
    s = 0.1
    alpha = 1.0
    # ax[0, 1].scatter(
    #     catalog_gamma["longitude"],
    #     catalog_gamma["latitude"],
    #     c=catalog_gamma["depth_km"],
    #     s=s,
    #     alpha=alpha,
    # )
    # ax[1, 1].scatter(
    #     catalog_gamma["longitude"],
    #     -catalog_gamma["depth_km"],
    #     c=catalog_gamma["depth_km"],
    #     s=s,
    #     alpha=alpha,
    # )
    # ax[2, 1].scatter(
    #     catalog_gamma["latitude"],
    #     -catalog_gamma["depth_km"],
    #     c=catalog_gamma["depth_km"],
    #     s=s,
    #     alpha=alpha,
    # )
    # xlim = ax[0, 1].get_xlim()
    # ylim = ax[0, 1].get_ylim()
    # zlim = ax[1, 1].get_ylim()
    # vmin = catalog_gamma["depth_km"].min()
    # vmax = catalog_gamma["depth_km"].max()
    # ax[1, 1].set_xlim(xlim)
    # ax[2, 1].set_xlim(ylim)

    ax[0, 0].scatter(
        stations["longitude"],
        stations["latitude"],
        c="k",
        s=10,
        alpha=0.3,
        marker="^",
    )
    ax[0, 0].scatter(
        catalog["longitude"],
        catalog["latitude"],
        # c=catalog["depth_km"],
        c="b",
        s=s,
        alpha=alpha,
    )
    ax[0,0].set_title(f"Number of events: {len(catalog)}")
    ax[1, 0].scatter(
        catalog["longitude"],
        catalog["depth_km"],
        # c=catalog["depth_km"],
        c="b",
        s=s,
        alpha=alpha,
    )
    ax[2, 0].scatter(
        catalog["latitude"],
        catalog["depth_km"],
        # c=catalog["depth_km"],
        c="b",
        s=s,
        alpha=alpha,
    )

    xlim_deg = [min_longitude, max_longitude]
    ylim_deg = [min_latitude, max_latitude]
    ax[0, 0].set_xlim(xlim_deg)
    ax[0, 0].set_ylim(ylim_deg)
    ax[1, 0].set_xlim(xlim_deg)
    ax[1, 0].set_ylim([zlim[0]-shift_z*2, zlim[1]])
    ax[2, 0].set_xlim(ylim_deg)
    ax[2, 0].set_ylim([zlim[0]-shift_z*2, zlim[1]])
    ax[1, 0].invert_yaxis()
    ax[2, 0].invert_yaxis()
    plt.savefig(f"location_{args.device}.png", dpi=300)

# %%
