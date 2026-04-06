import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import ocnn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
import src.models.ocnn_model_ref.my_ocnn as ocnn_unet


class AcousticFieldHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, pe_frequencies=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pe_frequencies = pe_frequencies
        self.register_buffer(
            "frequency_bands",
            (2.0 ** torch.arange(pe_frequencies, dtype=torch.float32)) * torch.pi,
            persistent=False,
        )
        input_dim = hidden_dim + 3 + 3 * 2 * pe_frequencies
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def positional_encoding(self, xyz):
        scaled_xyz = xyz.unsqueeze(-1) * self.frequency_bands.view(1, 1, -1)
        pe = torch.cat([scaled_xyz.sin(), scaled_xyz.cos()], dim=-1).reshape(xyz.size(0), -1)
        return torch.cat([xyz, pe], dim=-1)

    def forward(self, point_features, xyz):
        fused_features = torch.cat([point_features, self.positional_encoding(xyz)], dim=-1)
        return self.layers(fused_features)


class MyPipeline(pl.LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate if learning_rate is not None else getattr(cfg, "LEARNING_RATE", 1e-3)
        self.hidden_dim = getattr(cfg, "HIDDEN_DIM", 256)
        self.output_dim = getattr(cfg, "OUTPUT_DIM", 256)
        self.train_vis_every_n_epochs = max(1, int(getattr(cfg, "TRAIN_VIS_EVERY_N_EPOCHS", 1)))
        self.input_feature = ocnn.modules.InputFeature("NPD", nempty=cfg.OCTREE_NEMPTY)
        self.backbone_network = ocnn_unet.UNet(in_channels=7, out_channels=self.hidden_dim, nempty=cfg.OCTREE_NEMPTY)
        self.acoustic_head = AcousticFieldHead(self.hidden_dim, self.output_dim)

    def build_targets(self, batch_data):
        targets = torch.cat([mel.float().to(self.device).max(dim=-1).values for mel in batch_data["mel_spectrogram"]], dim=0)
        return F.adaptive_avg_pool1d(targets.unsqueeze(1), self.output_dim).squeeze(1)

    def compute_loss_terms(self, output, targets):
        smooth_l1_loss = F.smooth_l1_loss(output, targets)
        pred_dist = F.softplus(output)
        target_dist = targets.clamp_min(0)
        pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pred_cdf = torch.cumsum(pred_dist, dim=-1)
        target_cdf = torch.cumsum(target_dist, dim=-1)
        emd_loss = (pred_cdf - target_cdf).abs().mean()
        total_loss = smooth_l1_loss * 1 + emd_loss * 1
        return total_loss, smooth_l1_loss, emd_loss

    def build_prediction_report(self, batch_data, targets, output, loss, stage):
        gt = targets.detach().cpu()
        pred = output.detach().cpu()
        diff = (pred - gt).abs()
        object_impact_count = int(batch_data["num_impacts"][0].item())
        gt_object = gt[:object_impact_count]
        pred_object = pred[:object_impact_count]
        diff_object = diff[:object_impact_count]
        sample_idx = 0
        gt_sample = gt_object[sample_idx]
        pred_sample = pred_object[sample_idx]
        diff_sample = diff_object[sample_idx]
        sample_count = min(8, gt_object.size(0))
        gt_panel = gt_object[:sample_count]
        pred_panel = pred_object[:sample_count]
        mae = diff_object.mean().item()
        rmse = torch.sqrt(((pred_object - gt_object) ** 2).mean()).item()
        corr = torch.corrcoef(torch.stack([gt_sample, pred_sample]))[0, 1].item() if gt_sample.numel() > 1 else 0.0
        worst_dims = torch.topk(diff_sample, k=min(8, diff_sample.numel())).indices.tolist()
        impact_points = batch_data["impact_point"][0].detach().cpu()
        highlighted_point = impact_points[sample_idx]
        axis_variance = impact_points.var(dim=0)
        axis_order = torch.argsort(axis_variance, descending=True).tolist()
        axis_x = axis_order[0]
        axis_y = axis_order[1]
        axis_names = ["x", "y", "z"]
        obj_id = batch_data["obj_id"][0]

        fig = plt.figure(figsize=(16, 10), dpi=160)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.6, 1.6])

        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        text = "\n".join([
            f"epoch={self.current_epoch}  {stage}_loss={float(loss.item()):.6f}  obj_id={obj_id}  obj_impacts={object_impact_count}  dims={gt_object.size(1)}",
            f"object_mae={mae:.6f}  object_rmse={rmse:.6f}  impact_idx={sample_idx}  sample_corr={corr:.6f}",
            f"gt(mean/std/min/max)=({gt_sample.mean():.4f}, {gt_sample.std():.4f}, {gt_sample.min():.4f}, {gt_sample.max():.4f})",
            f"pred(mean/std/min/max)=({pred_sample.mean():.4f}, {pred_sample.std():.4f}, {pred_sample.min():.4f}, {pred_sample.max():.4f})",
            f"impact_xyz=({highlighted_point[0]:.4f}, {highlighted_point[1]:.4f}, {highlighted_point[2]:.4f})",
            f"worst_dims={worst_dims}",
        ])
        ax_text.text(0.01, 0.98, text, va="top", ha="left", family="monospace", fontsize=11)

        ax_line = fig.add_subplot(gs[1, 0])
        ax_line.plot(gt_sample.numpy(), label="GT", linewidth=2)
        ax_line.plot(pred_sample.numpy(), label="Pred", linewidth=2)
        ax_line.set_title("GT vs Pred")
        ax_line.legend()

        ax_diff = fig.add_subplot(gs[1, 1])
        ax_diff.bar(range(diff_sample.numel()), diff_sample.numpy(), color="tab:red")
        ax_diff.set_title("Absolute Error")

        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.scatter(
            impact_points[:, axis_x].numpy(),
            impact_points[:, axis_y].numpy(),
            s=24,
            alpha=0.7,
            label="Impacts",
        )
        ax_scatter.scatter(
            highlighted_point[axis_x].item(),
            highlighted_point[axis_y].item(),
            s=70,
            color="tab:red",
            label="Selected",
        )
        ax_scatter.set_title("Impact Positions")
        ax_scatter.set_xlabel(axis_names[axis_x])
        ax_scatter.set_ylabel(axis_names[axis_y])
        ax_scatter.legend()

        vmin = min(gt_panel.min().item(), pred_panel.min().item())
        vmax = max(gt_panel.max().item(), pred_panel.max().item())

        ax_gt = fig.add_subplot(gs[2, 0])
        im_gt = ax_gt.imshow(gt_panel.numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_gt.set_title("GT Heatmap")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        ax_pred = fig.add_subplot(gs[2, 1])
        im_pred = ax_pred.imshow(pred_panel.numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_pred.set_title("Pred Heatmap")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        ax_err = fig.add_subplot(gs[2, 2])
        im_err = ax_err.imshow((pred_panel - gt_panel).abs().numpy(), aspect="auto", cmap="magma")
        ax_err.set_title("AbsDiff Heatmap")
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.canvas.draw()
        image = torch.from_numpy(np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3]).permute(2, 0, 1)
        plt.close(fig)
        return image

    def forward(self, batch_data):
        impact_points = [points.to(self.device) for points in batch_data["impact_point"]]
        octree = batch_data["octree"].to(self.device)
        data = self.input_feature(octree)
        targets = self.build_targets(batch_data)
        point_xyz = torch.cat(impact_points, dim=0)
        impact_counts = torch.tensor([points.size(0) for points in impact_points], dtype=torch.long, device=self.device)
        query_batch_index = torch.repeat_interleave(
            torch.arange(len(impact_points), device=self.device, dtype=torch.long),
            impact_counts,
        )
        query_pts = torch.cat([point_xyz, query_batch_index[:, None].float()], dim=1)
        point_features = self.backbone_network(data=data, octree=octree, depth=octree.depth, query_pts=query_pts)
        output = self.acoustic_head(point_features, point_xyz)
        loss, smooth_l1_loss, emd_loss = self.compute_loss_terms(output, targets)
        return loss, output, smooth_l1_loss, emd_loss

    def training_step(self, batch, batch_idx):
        loss, output, smooth_l1_loss, emd_loss = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        self.log("train_smooth_l1_loss", smooth_l1_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("train_emd_loss", emd_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        
        opt = self.optimizers()
        if opt:
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
            
        should_log_train_visualization = (self.current_epoch + 1) % self.train_vis_every_n_epochs == 0
        if batch_idx == 0 and should_log_train_visualization and getattr(self.logger, "experiment", None) is not None:
            targets = self.build_targets(batch)
            report = self.build_prediction_report(batch, targets, output, loss, stage="train")
            self.logger.experiment.add_image("train/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, smooth_l1_loss, emd_loss = self(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        self.log("val_smooth_l1_loss", smooth_l1_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("val_emd_loss", emd_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        if batch_idx == 0 and getattr(self.logger, "experiment", None) is not None:
            targets = self.build_targets(batch)
            report = self.build_prediction_report(batch, targets, output, loss, stage="val")
            self.logger.experiment.add_image("val/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, smooth_l1_loss, emd_loss = self(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        self.log("test_smooth_l1_loss", smooth_l1_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        self.log("test_emd_loss", emd_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["num_impacts"].sum().item())
        return loss

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0),
        )
        total_epochs = max(1, int(getattr(cfg, "MAX_EPOCHS", 1)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / total_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
