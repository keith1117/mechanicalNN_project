import numpy as np
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
from os import path
from FE import StructuralFE
import matplotlib.pyplot as plt
from matplotlib import colors
from pytictoc import TicToc
import os.path as osp
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from TO_models import TopNet
from utils import PytorchMinMaxScaler, plot_latent, setDevice, set_seed
import utils
from material_models import MaterialModel
from material_models import elasticity
from utils import Logger
import copy

print("RUNNING FILE:", osp.abspath(__file__))

from matplotlib import rc
rc('text', usetex=False)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 150
timer = TicToc()

overrideGPU = False
device = setDevice(overrideGPU)
torch.autograd.set_detect_anomaly(True)


class TopologyOptimizer:
    def __init__(self, config):
        self.nelx = config.nelx
        self.nely = config.nely
        self.len_x = config.len_x
        self.len_y = config.len_y
        self.max_grad = config.max_grad
        self.simplexDim = config.simplexDim
        self.cell_width = config.cell_width
        self.cell_type = config.cell_type
        self.results_dir = config.results_dir
        self.interactive = config.interactive
        self.desiredVolumeFraction = config.desiredVolumeFraction

        # New target settings from config/app
        self.target_type = getattr(config, "target_type", "x")
        self.target_square_start = getattr(config, "target_square_start", 5)
        self.target_square_end = getattr(config, "target_square_end", 15)

        self.selecting_loading(config.example)
        self.exper_name = (
            self.exampleName + "_" + config.nn_type + "_" + config.cell_type + "_" + str(config.desiredVolumeFraction)
        )
        self.initializeFE(config)
        self.initializeOptimizer(config)
        self.InitializeMaterialModel(config, device)

    def initializeFE(self, config):
        self.FE = StructuralFE()
        self.FE.initializeSolver(
            config.nelx, config.nely, self.force, self.fixed, config.penal, config.Emin, config.Emax
        )
        self.xy, self.nonDesignIdx = self.generatePoints(config.nelx, config.nely, 1, self.nonDesignRegion)
        self.xyPlot, self.nonDesignPlotIdx = self.generatePoints(
            config.nelx, config.nely, config.cell_width, self.nonDesignRegion
        )

    def initializeOptimizer(self, config):
        self.density = config.desiredVolumeFraction * np.ones((self.nelx * self.nely))
        self.topNet = TopNet(config, self.symXAxis, self.symYAxis).to(device)
        self.objective = 0.0
        self.convergenceHistory = []

    def InitializeMaterialModel(self, config, device):
        self.material_model = MaterialModel(config, device)

    def compute_gradient_norm(self, nn_t):
        nn_t_matrix = nn_t.reshape(self.nelx, self.nely, self.simplexDim + 1)
        dx = self.len_x / self.nelx
        dy = self.len_y / self.nely
        ddx = (nn_t_matrix[1:, :, :] - nn_t_matrix[:-1, :, :]) / dx
        ddy = (nn_t_matrix[:, 1:, :] - nn_t_matrix[:, :-1, :]) / dy
        grad_norm = torch.sqrt(ddx[:, 1:, :] ** 2 + ddy[1:, :, :] ** 2)
        return torch.max(torch.flatten(grad_norm))

    def build_target_u(self):
        target_u = torch.zeros((self.nely, self.nelx), device=device)

        if self.target_type == "x":
            n = min(self.nelx, self.nely)
            for i in range(n):
                target_u[i, i] = 1
                target_u[i, n - 1 - i] = 1

        elif self.target_type == "square":
            r0 = max(0, int(self.target_square_start))
            r1 = min(self.nely, int(self.target_square_end))
            c0 = max(0, int(self.target_square_start))
            c1 = min(self.nelx, int(self.target_square_end))
            target_u[r0:r1, c0:c1] = 1.0

        elif self.target_type == "circle":
            center_x = self.nelx / 2
            center_y = self.nely / 2
            radius = min(self.nelx, self.nely) / 3
            Y, X = torch.meshgrid(
                torch.arange(0, self.nely, device=device),
                torch.arange(0, self.nelx, device=device),
                indexing='ij'
            )
            dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            target_u = 1 - torch.clamp(dist_from_center / radius, 0, 1)

        else:
            n = min(self.nelx, self.nely)
            for i in range(n):
                target_u[i, i] = 1
                target_u[i, n - 1 - i] = 1

        return target_u

    def optimizeDesign(self, config):
        train_logger = Logger(
            osp.join(config.results_dir, self.exper_name + 'train.log'),
            ['ep', 'compliance', 'real_compliance']
        )
        self.convergenceHistory = []
        savedNetFileName = osp.join(
            config.results_dir, self.exampleName + '_' + str(self.nelx) + '_' + str(self.nely) + '.nt'
        )
        savedMaterialNetFileName = osp.join(
            config.results_dir, self.exampleName + '_' + str(self.nelx) + '_' + str(self.nely) + 'material.nt'
        )
        alphaMax = 100 * config.desiredVolumeFraction
        alphaIncrement = 0.08
        alpha = alphaIncrement
        nrmThreshold = 0.1

        if config.useSavedNet:
            if path.exists(savedNetFileName):
                import sys
                # FIX: correct module key ('__main__' not 'main') and
                # correct attribute name (PytorchMinMaxScaler not PyTorchMinMaxScalar)
                sys.modules['__main__'].PytorchMinMaxScaler = utils.PytorchMinMaxScaler

                self.topNet = torch.load(savedNetFileName)
                self.material_model = torch.load(savedMaterialNetFileName)
            else:
                print("Network file not found")

        if config.nn_type == 'SIMP':
            self.optimizer = torch.optim.Adam([
                {'params': self.topNet.model.rho, 'lr': config.learningRate},
                {'params': self.topNet.model.t, 'lr': config.learningRate}
            ])
        else:
            self.optimizer = optim.Adam(self.topNet.parameters(), lr=config.learningRate)

        w = self.cell_width
        batch_x = self.xy.view(-1, 2).float().to(device)

        # New configurable target
        target_u = self.build_target_u()
        print(f"RUNNING TARGET TYPE: {self.target_type}")

        nn_rho = torch.ones(self.nelx * self.nely).to(device)

        for epoch in range(config.maxEpochs):
            self.optimizer.zero_grad()
            _, nn_t = self.topNet(batch_x, 1, self.nonDesignIdx)
            interpolate_list, nn_C, v = self.material_model.map2material(nn_t)
            true_v = torch.sum(torch.sum(interpolate_list, dim=2), dim=1) / (w * w)
            true_rho = nn_rho * true_v
            u, Jelem = self.FE.solvelatticetorch(nn_rho, nn_C)

            ux = Jelem.reshape((self.nelx, self.nely)).T
            max_u = torch.max(ux)
            min_u = torch.min(ux)
            u_normalized = (ux - min_u) / (max_u - min_u + 1e-8)
            u_normalized = torch.sigmoid(10 * (u_normalized - 0.5))

            nn_rho_vec = nn_rho.view(self.nelx, self.nely).T
            compliance = torch.mean((u_normalized - target_u) ** 2)
            grad_norm = self.compute_gradient_norm(nn_t)
            self.objective = compliance

            volConstraint = ((torch.mean(true_rho) / config.desiredVolumeFraction) - 1.0)
            currentVolumeFraction = torch.mean(true_rho).item()

            greyLoss = torch.sum((true_rho > 0.2) * (true_rho < 0.8)).float() / true_rho.shape[0]
            gradConstraint = 1 / (1 + torch.exp(-2 * (grad_norm - self.max_grad)))

            # loss = self.objective + alpha * (pow(volConstraint,2) + greyLoss)
            loss = self.objective
            alpha = min(alphaMax, alpha + alphaIncrement)

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            self.optimizer.step()

            if volConstraint < 0.05:
                greyElements = torch.sum((nn_rho > 0.05) * (nn_rho < 0.95)).item()
                relGreyElements = greyElements / nn_rho.shape[0]
            else:
                relGreyElements = 1

            self.convergenceHistory.append([
                self.objective.item(),
                currentVolumeFraction,
                loss.item(),
                relGreyElements
            ])

            self.FE.penal = min(4.0, self.FE.penal + 0.01)

            if epoch % 10 == 0:
                print(
                    "{:3d} J: {:.4F}; Vf: {:.3F}; GradNorm: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} ".format(
                        epoch,
                        self.objective.item(),
                        currentVolumeFraction,
                        grad_norm.item(),
                        loss.item(),
                        relGreyElements
                    )
                )

        self.plotTO(epoch, nn_rho, ux, interpolate_list, u_normalized, target_u, saveFig=True, saveFrame=config.saveFrame)

    def plotTO(self, iter, nn_rho, u_true, x_hat_list, u, target_u, saveFig=False, saveFrame=False):
        w = self.cell_width
        nn_rho = nn_rho.to('cpu').detach()
        x_hat_list = x_hat_list.to('cpu').detach().view(-1, w, w)
        true_v = torch.sum(torch.sum(x_hat_list, dim=2), dim=1) / (w * w)
        true_rho = nn_rho * true_v

        nn_rho_np = nn_rho.numpy()
        x_hat_list_np = x_hat_list.detach().cpu().numpy()
        img = np.zeros(((self.FE.nely) * w, (self.FE.nelx) * w))
        for i, x_hat in enumerate(x_hat_list_np):
            block_x = self.FE.nely - i % self.FE.nely
            block_y = i // self.FE.nely
            img[(block_x - 1) * w:block_x * w, block_y * w:(block_y + 1) * w] = np.flip(x_hat.transpose(), axis=0) * nn_rho_np[i]
        img = np.flip(img.transpose(), axis=0)

        if self.interactive:
            plt.ion()
        plt.clf()

        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        axes = plt.gca()
        cmap = plt.get_cmap('Greens')
        axes.imshow(img, cmap=cmap, vmin=0, vmax=1)
        fName = osp.join(self.results_dir, self.exper_name + '_topology.jpg')
        plt.savefig(fName, dpi=450, transparent=False)

        data_file_name = osp.join(self.results_dir, self.exper_name + '_img.npy')
        np.save(data_file_name, img, allow_pickle=False)

        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        axes = plt.gca()
        u = u.to('cpu').detach().numpy()
        u = np.flip(u.transpose(), axis=0)
        cmap = plt.get_cmap('jet')
        im = axes.imshow(u, cmap=cmap)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        fName = osp.join(self.results_dir, self.exper_name + '_normalized_displacement.jpg')
        plt.savefig(fName, dpi=450, transparent=False)

        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        axes = plt.gca()
        u_true = u_true.to('cpu').detach().numpy()
        u_true = np.flip(u_true.transpose(), axis=0)
        cmap = plt.get_cmap('jet')
        im = axes.imshow(u_true, cmap=cmap)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        fName = osp.join(self.results_dir, self.exper_name + '_true_displacement.jpg')
        plt.savefig(fName, dpi=450, transparent=False)

        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        axes = plt.gca()
        target_u = target_u.to('cpu').detach().numpy()
        target_u = np.flip(target_u.transpose(), axis=0)
        cmap = plt.get_cmap('jet')
        im = axes.imshow(target_u, cmap=cmap)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        fName = osp.join(self.results_dir, self.exper_name + '_target_displacement.jpg')
        plt.savefig(fName, dpi=450, transparent=False)

    def plotTO_smooth(self, iter, saveFig=False):
        saveFrame = False
        w = self.cell_width
        batch_x = self.xy.view(-1, 2).float().to(device)
        nn_rho, nn_t = self.topNet(batch_x, 1, self.nonDesignIdx)
        nn_rho = nn_rho.to('cpu').detach().numpy()

        if self.cell_type == "lattice":
            interpolate_list, nn_C, v = self.material_model.map2material(nn_t)
            interpolate_list_np = interpolate_list.detach().numpy()
            true_v = np.sum(np.sum(interpolate_list_np, axis=2), axis=1) / (w * w)
            true_rho = nn_rho * true_v
            lattice_img = np.zeros(((self.FE.nely) * w, (self.FE.nelx) * w))
            solid_img = np.zeros(((self.FE.nely) * w, (self.FE.nelx) * w))
            for i, x_hat in enumerate(interpolate_list_np):
                block_x = self.FE.nely - i % self.FE.nely
                block_y = i // self.FE.nely
                lattice_img[(block_x - 1) * w:block_x * w, block_y * w:(block_y + 1) * w] = np.flip(x_hat.transpose(), axis=0) * nn_rho[i]
                solid_img[(block_x - 1) * w:block_x * w, block_y * w:(block_y + 1) * w] = np.ones(x_hat.shape) * nn_rho[i]
            img = lattice_img
        else:
            solid_img = np.flip(nn_rho.reshape(self.FE.nelx, self.FE.nely).transpose(), axis=0)
            true_rho = nn_rho
            img = solid_img

        large_batch_x = self.xyPlot.view(-1, 2).float().to(device)
        large_nn_rho, large_nn_t = self.topNet(large_batch_x, w, self.nonDesignPlotIdx)
        large_nn_rho = large_nn_rho.to('cpu').detach().numpy()
        large_img = np.flip(large_nn_rho.reshape(self.FE.nelx * w, self.FE.nely * w).transpose(), axis=0)
        large_img[large_img > 0.1] = 1
        large_img[large_img < 0.1] = 0

        if self.cell_type == "lattice":
            solid_img[solid_img > 0.5] = 1
            solid_img[solid_img < 0.5] = 0
            lattice_img[lattice_img > 0.5] = 1
            lattice_img[lattice_img < 0.5] = 0
            large_img = large_img - solid_img + lattice_img

        img = large_img

        if self.interactive:
            plt.ion()

        plt.clf()
        if saveFig:
            example = self.example
            if self.cell_type == "lattice":
                real_compliance, dist = self.full_structure_FE(example, self.cell_type, nn_rho, w, interpolate_list)
            else:
                real_compliance, dist = self.full_structure_FE(example, self.cell_type, nn_rho, w)
            print("real compliance is: {}".format(real_compliance))
        else:
            real_compliance = self.objective * self.obj0

        plt.xticks([])
        plt.yticks([])
        plt.title(
            'Iter = {:d}, J = {:.2F}, V_f = {:.2F}, V_des = {:.2F}'.format(
                iter, real_compliance, np.mean(true_rho), self.desiredVolumeFraction
            ),
            loc='left'
        )
        plt.grid(False)
        axes = plt.gca()
        cmap = 'Greens'
        cmap = plt.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cbar = plt.colorbar(m, cax=cax, aspect=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Density", fontsize=10)
        plt.ticklabel_format(style="plain")
        axes.imshow(img, cmap=cmap, vmin=0, vmax=1)

        if saveFrame:
            frame_file_name = osp.join(self.results_dir, 'frames', 'f_' + '+str(iter)' + '.jpg')
            plt.savefig(frame_file_name)

        if saveFig:
            fName = osp.join(self.results_dir, self.exampleName + '_topology.png')
            plt.savefig(fName, dpi=450)
            data_file_nme = osp.join(self.results_dir, self.exampleName + '_img.npy')
            np.save(data_file_nme, img, allow_pickle=False)

        if self.interactive:
            plt.pause(0.01)

    def plotConvergence(self):
        self.convergenceHistory = np.array(self.convergenceHistory)
        plt.figure()
        plt.semilogy(self.convergenceHistory[:, 0], 'b:', label='Rel. Compliance')
        plt.semilogy(self.convergenceHistory[:, 1], 'r--', label='Vol. Fraction')
        plt.title('Convergence Plots')
        plt.title('Convergence plots; V_des = {:.2F}'.format(self.desiredVolumeFraction))
        plt.xlabel('Iterations')
        plt.grid('True')
        plt.legend(loc='lower left', shadow=True, fontsize='large')
        fName = osp.join(self.results_dir, self.exper_name + '_convergence.png')
        plt.savefig(fName, dpi=450)

    def setup_structuralFE(self, example, nelx, nely, penal):
        large_ndof = 2 * (nelx + 1) * (nely + 1)
        large_force = np.zeros((large_ndof, 1))
        large_dofs = np.arange(large_ndof)

        if example == 1:
            large_fixed = large_dofs[0:2 * (nely + 1):1]
            large_force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
            loading_point = 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1
        if example == 2:
            large_fixed = large_dofs[0:2 * (nely + 1):1]
            large_force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
            loading_point = 2 * (nelx + 1) * (nely + 1) - (nely + 1)
        if example == 3:
            large_fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1)
            large_force[2 * (nely + 1) + 1, 0] = -1
            loading_point = 2 * (nely + 1) + 1
        if example == 4:
            large_fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely])
            large_force[nelx * (nely + 1) + 1, 0] = -1
            loading_point = nelx * (nely + 1) + 1
        if example == 5:
            large_fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely])
            large_force[2 * nely + 1:2 * (nelx + 1) * (nely + 1):2 * (nely + 1), 0] = -1 / (nelx + 1)
            loading_point = 2 * (nelx + 1) * (nely + 1) - 1
        if example == 6:
            large_fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 1)
            large_force[2 * (nelx + 1) * (nely + 1) - (nely), 0] = 1
            loading_point = 2 * (nelx + 1) * (nely + 1) - (nely)
        if example == 7:
            large_fixed = large_dofs[0:2 * (nely + 1):1]
            large_force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
            large_force[2 * (nelx + 1) * (nely + 1) - 2, 0] = 1
            loading_point = 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1
        if example == 8:
            large_fixed = large_dofs[0:2 * (nely + 1):1]
            large_force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
            large_force[2 * (nelx + 1) * (nely + 1) - 2, 0] = 1
            loading_point = 2 * (nelx + 1) * (nely + 1) - (nely + 1)
        if example == 9:
            large_fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1)
            large_force[2 * (nely + 1) + 1, 0] = -1
            large_force[2 * (nelx + 1) * (nely + 1) - 2, 0] = 1
            loading_point = 2 * (nely + 1) + 1

        FE_solver = StructuralFE()
        FE_solver.initializeSolver(nelx, nely, large_force, large_fixed, penal, Emin=1e-6, Emax=1.0)
        FE_solver.loading_point = loading_point
        return FE_solver

    def full_structure_FE(self, example, cell_type, nn_rho, w, interpolate_list=None):
        img = torch.zeros(((self.FE.nely) * w, (self.FE.nelx) * w))
        for i in range(nn_rho.shape[0]):
            block_x = self.FE.nely - i % self.FE.nely
            block_y = i // self.FE.nely
            if cell_type == "lattice":
                x_hat = interpolate_list[i]
                img[(block_x - 1) * w:block_x * w, block_y * w:(block_y + 1) * w] = torch.flip(torch.transpose(x_hat, 0, 1), dims=(0,)) * nn_rho[i]
            else:
                img[(block_x - 1) * w:block_x * w, block_y * w:(block_y + 1) * w] = torch.ones((w, w)) * nn_rho[i]
        large_rho = torch.transpose(torch.flip(img, dims=(0,)), 0, 1).reshape((self.FE.nelx * self.FE.nely * w * w)).cpu().detach().numpy()

        large_FE = self.setup_structuralFE(example, self.FE.nelx * w, self.FE.nely * w, self.FE.penal)
        large_u, large_Jelem = large_FE.solve88(large_rho)
        compliance = np.sum((0.01 + (large_rho ** 2) * 0.99) * large_Jelem)
        disp_force = large_u[large_FE.loading_point]
        return compliance, disp_force

    def selecting_loading(self, example):
        nelx = self.nelx
        nely = self.nely
        self.example = example

        if example == 1:
            self.exampleName = 'TipCantilever'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = self.dofs[0:2 * (nely + 1):1]
            self.force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = False
            self.symYAxis = False

        elif example == 2:
            self.exampleName = 'MidCantilever'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = self.dofs[0:2 * (nely + 1):1]
            self.force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = True
            self.symYAxis = False

        elif example == 3:
            self.exampleName = 'MBBBeam'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1)
            self.force[2 * (nely + 1) + 1, 0] = -1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = False
            self.symYAxis = False

        elif example == 4:
            self.exampleName = 'Michell'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely])
            self.force[nelx * (nely + 1) + 1, 0] = -1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': {'center': [30., 15.], 'rad_out': 6., 'rad_in': 3}}
            self.symXAxis = False
            self.symYAxis = True

        elif example == 5:
            self.exampleName = 'Bridge'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely])
            self.force[2 * nely + 1:2 * (nelx + 1) * (nely + 1):2 * (nely + 1), 0] = -1 / (nelx + 1)
            self.nonDesignRegion = {'Rect': {'x>': 0, 'x<': nelx, 'y>': nely - 1, 'y<': nely}, 'Circ': None, 'Annular': None}
            self.symXAxis = False
            self.symYAxis = True

        elif example == 6:
            self.exampleName = 'TensileBar'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 1)
            self.midDofX = 2 * (nelx + 1) * (nely + 1) - (nely)
            self.force[self.midDofX, 0] = 1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = True
            self.symYAxis = False

        elif example == 7:
            self.exampleName = 'ComplexCantilever'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = self.dofs[0:2 * (nely + 1):1]
            self.force[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0] = -1
            self.force[2 * (nelx + 1) * (nely + 1) - 2, 0] = 1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = False
            self.symYAxis = False

        elif example == 8:
            self.exampleName = 'Mid_Complex_Cantilever'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = self.dofs[0:2 * (nely + 1):1]
            self.force[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0] = -1
            self.force[2 * (nelx + 1) * (nely + 1) - 2, 0] = 1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = False
            self.symYAxis = False

        elif example == 9:
            self.exampleName = 'Complex_MBBBeam'
            self.ndof = 2 * (nelx + 1) * (nely + 1)
            self.force = np.zeros((self.ndof, 1))
            self.dofs = np.arange(self.ndof)
            self.fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), 2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1)
            self.force[2 * (nely + 1) + 1, 0] = -1
            self.force[2 * (nelx + 1) * (nely + 1) - 2, 0] = 1
            self.nonDesignRegion = {'Rect': None, 'Circ': None, 'Annular': None}
            self.symXAxis = False
            self.symYAxis = False

    def generatePoints(self, nelx, nely, resolution=1, nonDesignRegion=None):
        ctr = 0
        xy = np.zeros((resolution * nelx * resolution * nely, 2))
        nonDesignIdx = torch.zeros((resolution * nelx * resolution * nely), requires_grad=False).to(device)

        for i in range(resolution * nelx):
            for j in range(resolution * nely):
                xy[ctr, 0] = (i + 0.5) / resolution
                xy[ctr, 1] = (j + 0.5) / resolution

                if nonDesignRegion['Rect'] is not None:
                    if (
                        (xy[ctr, 0] < nonDesignRegion['Rect']['x<']) and
                        (xy[ctr, 0] > nonDesignRegion['Rect']['x>']) and
                        (xy[ctr, 1] < nonDesignRegion['Rect']['y<']) and
                        (xy[ctr, 1] > nonDesignRegion['Rect']['y>'])
                    ):
                        nonDesignIdx[ctr] = 1

                if nonDesignRegion['Circ'] is not None:
                    if (
                        ((xy[ctr, 0] - nonDesignRegion['Circ']['center'][0]) ** 2 +
                         (xy[ctr, 1] - nonDesignRegion['Circ']['center'][1]) ** 2)
                        <= nonDesignRegion['Circ']['rad'] ** 2
                    ):
                        nonDesignIdx[ctr] = 1

                if nonDesignRegion['Annular'] is not None:
                    locn = (
                        (xy[ctr, 0] - nonDesignRegion['Annular']['center'][0]) ** 2 +
                        (xy[ctr, 1] - nonDesignRegion['Annular']['center'][1]) ** 2
                    )
                    if (
                        (locn <= nonDesignRegion['Annular']['rad_out'] ** 2) and
                        (locn > nonDesignRegion['Annular']['rad_in'] ** 2)
                    ):
                        nonDesignIdx[ctr] = 1

                ctr += 1

        xy = torch.tensor(xy, requires_grad=True).float().view(-1, 2).to(device)
        return xy, nonDesignIdx