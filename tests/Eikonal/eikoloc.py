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

multiprocessing.set_start_method("spawn", True)

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

# %%
def traveltime(event_loc, station_locs, time_table, rgrid, zgrid, h, sigma=1, bounds=None):

    event_loc = event_loc.unsqueeze(0)
    r = torch.sqrt(torch.sum((event_loc[:, :2] - station_locs[:, :2]) ** 2, dim=-1))
    z = torch.abs(event_loc[:, 2] - station_locs[:, 2])

    r = r.unsqueeze(-1).unsqueeze(-1)
    z = z.unsqueeze(-1).unsqueeze(-1)

    magn = (
        1.0
        / (2.0 * np.pi * sigma)
        * torch.exp(-(((rgrid - r) / (np.sqrt(2 * sigma) * h)) ** 2 + ((zgrid - z) / (np.sqrt(2 * sigma) * h)) ** 2))
    )
    sum_magn = torch.sum(magn, dim=(-1, -2))
    tt = torch.sum(time_table * magn, dim=(-1, -2)) / sum_magn

    return tt


# %%
def invert(time, loc, type, weight, up, us, rgrid, zgrid, h, t0=[0], loc0=[0, 0, 0], device="cpu"):

    # device = up.device
    loc0 = torch.tensor(loc0, dtype=torch.float32, requires_grad=True, device=device)
    t0 = torch.tensor(t0, dtype=torch.float32, requires_grad=True, device=device)

    p_index = torch.arange(len(type), device=device)[type == "p"]
    s_index = torch.arange(len(type), device=device)[type == "s"]
    time_p = time[p_index]
    time_s = time[s_index]
    loc_p = loc[p_index]
    loc_s = loc[s_index]
    weight_p = weight[p_index]
    weight_s = weight[s_index]

    # %% optimization
    optimizer = optim.LBFGS(params=[t0, loc0], max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        if len(p_index) > 0:
            tt_p = t0 + traveltime(loc0, loc_p, up, rgrid, zgrid, h, sigma=1)
            loss_p = torch.mean(F.huber_loss(time_p, tt_p, reduction="none") * weight_p)
            # loss_p = F.mse_loss(time_p, tt_p)
        else:
            loss_p = 0
        if len(s_index) > 0:
            tt_s = t0 + traveltime(loc0, loc_s, us, rgrid, zgrid, h, sigma=1)
            loss_s = torch.mean(F.huber_loss(time_s, tt_s, reduction="none") * weight_s)
            # loss_s = F.mse_loss(time_s, tt_s)
        else:
            loss_s = 0
        loss = loss_p + loss_s
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)

    return t0.detach().cpu(), loc0.detach().cpu()


# %%
def run(catalog, event_index, picks, center, shift_z, up, us, rgrid, zgrid, h, device):

    if event_index == -1:
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
    event_t0, event_loc0 = invert(
        picks_time, picks_loc, picks_type, picks_weight, up, us, rgrid, zgrid, h, device=device
    )
    origin_time = picks_tmin + pd.to_timedelta(event_t0.item(), unit="s")
    longitude = event_loc0[0].item() / 111.2 + center[0]
    latitude = event_loc0[1].item() / 111.2 + center[1]
    depth = -event_loc0[2].item() + shift_z

    catalog.append(
        {
            "event_index": event_index,
            "time": origin_time,
            "latitude": latitude,
            "longitude": longitude,
            "depth_km": depth,
        }
    )
    if len(catalog) % 100 == 0:
        print(f"{len(catalog)} events are inverted.")


# %%

if __name__ == "__main__":

    ##
    device = "cuda"
    # device = "cpu"

    # %% Set domain
    xlim = [0, 100]
    ylim = [0, 100]
    zlim = [0, 30]  ## depth
    h = 1.0

    rlim = [0, ((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2) ** 0.5]

    rgrid = np.linspace(rlim[0], rlim[1], round((rlim[1] - rlim[0]) / h))
    zgrid = np.linspace(zlim[0], zlim[1], round((zlim[1] - zlim[0]) / h))
    m = len(rgrid)
    n = len(zgrid)
    dr = h
    dz = h

    # %% Calculate traveltime table
    x = [0, 5.5, 5.5, 16.0, 16.0, 32.0, 32.0]
    vp = [5.5, 5.5, 6.3, 6.3, 6.7, 6.7, 7.8]
    vp1d = np.interp(zgrid, x, vp)
    vs1d = vp1d / 1.73
    plt.figure()
    plt.plot(x, vp, "o")
    plt.plot(zgrid, vp1d)
    plt.plot(zgrid, vs1d)
    plt.savefig("velocity_model.png")

    # %%

    vp = np.ones((m, n)) * vp1d
    vs = np.ones((m, n)) * vs1d

    up = 1000 * np.ones((m, n))
    up[0, 0] = 0.0
    up = eikonal_solve(up, vp, h)

    us = 1000 * np.ones((m, n))
    us[0, 0] = 0.0
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
    stations = pd.read_csv("stations.csv", sep="\t")
    stations["id"] = stations["station"]
    picks = picks.merge(stations, on="id")

    # %% Prepare input data

    center = [stations["longitude"].mean(), stations["latitude"].mean(), 0]
    shift_z = picks["elevation(m)"].max() / 1000
    picks["x_km"] = (picks["longitude"] - center[0]) * 111.2 * np.cos(np.deg2rad(center[1]))
    picks["y_km"] = (picks["latitude"] - center[1]) * 111.2
    picks["z_km"] = -picks["elevation(m)"] / 1000 + shift_z

    # %% sequential inversion
    # catalog = []
    # num = 0
    # for event_index in tqdm(list(picks["event_idx"].unique())):
    #     if event_index == -1:
    #         continue

    #     picks_ = picks[picks["event_idx"] == event_index]

    #     # %%
    #     picks_tmin = picks_["timestamp"].min()
    #     picks_tt = (picks_["timestamp"] - picks_tmin).dt.total_seconds()
    #     picks_time = torch.tensor(picks_tt.values, dtype=torch.float32)
    #     picks_loc = torch.tensor(picks_[["x_km", "y_km", "z_km"]].values, dtype=torch.float32)
    #     picks_type = picks_["type"].values
    #     picks_weight = torch.tensor(picks_["prob"].values, dtype=torch.float32)

    #     # %%
    #     event_t0, event_loc0 = invert(picks_time, picks_loc, picks_type, picks_weight, up, us, rgrid, zgrid, h, device=device)
    #     origin_time = picks_tmin + pd.to_timedelta(event_t0.item(), unit="s")
    #     longitude = event_loc0[0].item()/111.2 + center[0]
    #     latitude = event_loc0[1].item()/ 111.2 + center[1]
    #     depth = - event_loc0[2].item() + shift_z

    #     catalog.append({"event_index": event_index, "time": origin_time, "latitude": latitude, "longitude": longitude, "depth_km": depth})

    #     num += 1
    #     #if num > 20:
    #     #    break

    # %% parallel inversion
    mangaer = mp.Manager()
    catalog = mangaer.list()
    # num_cpu = mp.cpu_count() // 4
    num_cpu = 8
    num_gpu = torch.cuda.device_count()
    print("num_cpu", num_cpu)
    print(f"Total number of events: {len(picks['event_idx'].unique())}")

    with mp.Pool(num_cpu) as pool:
        pool.starmap(
            run,
            [
                (
                    catalog,
                    event_index,
                    picks[picks["event_idx"] == event_index],
                    center,
                    shift_z,
                    up,
                    us,
                    rgrid,
                    zgrid,
                    h,
                    device + f":{i%num_gpu}",
                )
                for i, event_index in enumerate(list(picks["event_idx"].unique()))
            ],
        )

    catalog = pd.DataFrame(list(catalog))
    catalog.to_csv("catalog.csv", index=False)

    # %%
    catalog = pd.read_csv("catalog.csv", parse_dates=["time"])
    catalog_gamma = pd.read_csv("catalog_gamma.csv", sep="\t", parse_dates=["time"])
    catalog_gamma["depth_km"] = catalog_gamma["depth(m)"] / 1000

    fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(10, 8), gridspec_kw={"height_ratios": [4, 1, 1]})
    s = 0.1
    ax[0, 1].scatter(
        catalog_gamma["longitude"],
        catalog_gamma["latitude"],
        c=catalog_gamma["depth_km"],
        s=s,
    )
    ax[1, 1].scatter(
        catalog_gamma["longitude"],
        -catalog_gamma["depth_km"],
        c=catalog_gamma["depth_km"],
        s=s,
    )
    ax[2, 1].scatter(
        catalog_gamma["latitude"],
        -catalog_gamma["depth_km"],
        c=catalog_gamma["depth_km"],
        s=s,
    )
    xlim = ax[0, 1].get_xlim()
    ylim = ax[0, 1].get_ylim()
    zlim = ax[1, 1].get_ylim()
    ax[1, 1].set_xlim(xlim)
    ax[2, 1].set_xlim(ylim)
    ax[0, 0].scatter(
        catalog["longitude"],
        catalog["latitude"],
        c=catalog["depth_km"],
        s=s,
    )
    ax[1, 0].scatter(
        catalog["longitude"],
        -catalog["depth_km"],
        c=catalog["depth_km"],
        s=s,
    )
    ax[2, 0].scatter(
        catalog["latitude"],
        -catalog["depth_km"],
        c=catalog["depth_km"],
        s=s,
    )
    ax[0, 0].set_xlim(xlim)
    ax[0, 0].set_ylim(ylim)
    ax[1, 0].set_xlim(xlim)
    ax[1, 0].set_ylim(zlim)
    ax[2, 0].set_xlim(ylim)
    ax[2, 0].set_ylim(zlim)
    plt.savefig("location.png", dpi=300)

    raise


# %%
