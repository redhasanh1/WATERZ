import argparse
import os
from typing import Optional, Tuple

import ptlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

from RAFT import RAFT
from model.modules.flow_loss_utils import flow_warp, ternary_loss2


def initialize_RAFT(model_path='weights/raft-things.pth', device='cuda'):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model, map_location='cpu'))
    model = model.module

    model.to(device)

    return model

class FastFlowNetTRTEngine:
    """
    Thin TensorRT wrapper for the exported FastFlowNet engine.

    Falls back to raising RuntimeError if TensorRT Python libraries are missing.
    """

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt  # noqa: F401
            from cuda import cudart  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("TensorRT Python modules not available") from exc

        import tensorrt as trt
        from cuda import cudart

        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        # Create a dedicated CUDA stream
        error, stream = self._cudart.cudaStreamCreate()
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed with error {error}")
        self._stream = stream

        # Cache binding indices
        self._input_binding = None
        self._output_binding = None
        for idx in range(self._engine.num_bindings):
            if self._engine.binding_is_input(idx):
                self._input_binding = idx
            else:
                self._output_binding = idx

    def infer(self, images_np: "np.ndarray") -> "np.ndarray":
        import numpy as np

        cudart = self._cudart

        if self._input_binding is None or self._output_binding is None:
            raise RuntimeError("Invalid engine bindings detected")

        images_np = np.ascontiguousarray(images_np.astype(np.float32))
        self._context.set_binding_shape(self._input_binding, images_np.shape)
        output_shape = tuple(self._context.get_binding_shape(self._output_binding))
        output_np = np.empty(output_shape, dtype=np.float32)

        # Allocate device buffers
        error, d_input = cudart.cudaMalloc(images_np.nbytes)
        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc input failed with error {error}")

        error, d_output = cudart.cudaMalloc(output_np.nbytes)
        if error != cudart.cudaError_t.cudaSuccess:
            cudart.cudaFree(d_input)
            raise RuntimeError(f"cudaMalloc output failed with error {error}")

        try:
            cudart.cudaMemcpyAsync(
                d_input,
                images_np.ctypes.data,
                images_np.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )

            bindings = [0] * self._engine.num_bindings
            bindings[self._input_binding] = int(d_input)
            bindings[self._output_binding] = int(d_output)

            if not self._context.execute_async_v2(bindings, self._stream):
                raise RuntimeError("TensorRT execute_async_v2 returned False")

            cudart.cudaMemcpyAsync(
                output_np.ctypes.data,
                d_output,
                output_np.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )
            cudart.cudaStreamSynchronize(self._stream)
        finally:
            cudart.cudaFree(d_input)
            cudart.cudaFree(d_output)

        return output_np

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            if hasattr(self, "_cudart") and hasattr(self, "_stream"):
                self._cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass


# using fastflownet to replace raft with optional TensorRT acceleration
class RAFT_bi(nn.Module):
    """Flow completion loss"""

    def __init__(
        self,
        model_path: str = "weights/raft-things.pth",
        device: str = "cuda",
        *,
        trt_engine_path: Optional[str] = None,
    ):
        super().__init__()

        self._device = device
        self._trt_runner: Optional[FastFlowNetTRTEngine] = None

        if trt_engine_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            default_engine = os.path.abspath(
                os.path.join(module_dir, "..", "..", "engines", "raft", "raft_fp16.engine")
            )
            trt_engine_path = os.getenv("FASTFLOWNET_TRT_ENGINE", default_engine)

        trt_engine_path = (
            trt_engine_path if trt_engine_path and os.path.exists(trt_engine_path) else None
        )

        if trt_engine_path:
            try:
                self._trt_runner = FastFlowNetTRTEngine(trt_engine_path)
                print(
                    f"[FastFlowNet] Using TensorRT engine for flow estimation: {trt_engine_path}"
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                print(
                    f"[FastFlowNet] TensorRT engine load failed ({exc}); "
                    "falling back to PyTorch implementation."
                )
                self._trt_runner = None

        if self._trt_runner is None:
            self.fix_raft = ptlflow.get_model(
                "fastflownet", pretrained_ckpt="things"
            )
            self.fix_raft.to(device)
            for p in self.fix_raft.parameters():
                p.requires_grad = False

        self.l1_criterion = nn.L1Loss()
        self.eval()

    def _run_trt(
        self, frames_t: torch.Tensor, frames_tp1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        if self._trt_runner is None:
            raise RuntimeError("TensorRT runner is not initialized")

        pair_forward = (
            torch.stack((frames_t, frames_tp1), dim=0)
            .unsqueeze(0)
            .detach()
            .cpu()
            .numpy()
        )
        pair_backward = (
            torch.stack((frames_tp1, frames_t), dim=0)
            .unsqueeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        flows_forward = self._trt_runner.infer(pair_forward)
        flows_backward = self._trt_runner.infer(pair_backward)

        flows_forward_t = torch.from_numpy(np.asarray(flows_forward)).to(
            frames_t.device, dtype=frames_t.dtype
        )
        flows_backward_t = torch.from_numpy(np.asarray(flows_backward)).to(
            frames_t.device, dtype=frames_t.dtype
        )

        return flows_forward_t, flows_backward_t

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()

        if self._trt_runner is not None:
            flows_f = []
            flows_b = []
            for idx in range(l_t - 1):
                flow_f, flow_b = self._run_trt(
                    gt_local_frames[0, idx], gt_local_frames[0, idx + 1]
                )
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            gt_flows_forward = torch.stack(flows_f, dim=0).view(b, l_t - 1, 2, h, w)
            gt_flows_backward = torch.stack(flows_b, dim=0).view(b, l_t - 1, 2, h, w)
            return gt_flows_forward, gt_flows_backward

        with torch.no_grad():
            gtlf_1 = gt_local_frames[0, :-1, :, :, :]
            gtlf_2 = gt_local_frames[0, 1:, :, :, :]

            gt_flows_forward = self.fix_raft(
                {"images": torch.stack((gtlf_1, gtlf_2), dim=1)}
            )["flows"]
            gt_flows_backward = self.fix_raft(
                {"images": torch.stack((gtlf_2, gtlf_1), dim=1)}
            )["flows"]

        gt_flows_forward = gt_flows_forward.view(b, l_t - 1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t - 1, 2, h, w)

        return gt_flows_forward, gt_flows_backward


##################################################################################
def smoothness_loss(flow, cmask):
    delta_u, delta_v, mask = smoothness_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def smoothness_deltas(flow):
    """
    flow: [b, c, h, w]
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat((mask_x, mask_y), dim=1)
    mask = mask.to(flow.device)
    filter_x = torch.tensor([[0, 0, 0.], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 0, 0.], [0, 1, 0], [0, -1, 0]])
    weights = torch.ones([2, 1, 3, 3])
    weights[0, 0] = filter_x
    weights[1, 0] = filter_y
    weights = weights.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask


def second_order_loss(flow, cmask):
    delta_u, delta_v, mask = second_order_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x
    All positions where mask == 0 are not taken into account
    x: a tensor of shape [b, c, h, w]
    mask: a mask of shape [b, mc, h, w], where mask channels must be either 1 or the same as
    the number of channels of x. Entries should be 0 or 1
    return: loss
    """
    b, c, h, w = x.shape
    norm = b * c * h * w
    error = torch.pow(torch.square(x * beta) + torch.square(torch.tensor(epsilon)), alpha)
    if mask is not None:
        error = mask * error
    if truncate is not None:
        error = torch.min(error, truncate)
    return torch.sum(error) / norm


def second_order_deltas(flow):
    """
    consider the single flow first
    flow shape: [b, c, h, w]
    """
    # create mask
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)
    mask = mask.to(flow.device)

    filter_x = torch.tensor([[0, 0, 0.], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 1, 0.], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = torch.tensor([[1, 0, 0.], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = torch.tensor([[0, 0, 1.], [0, -2, 0], [1, 0, 0]])
    weights = torch.ones([4, 1, 3, 3])
    weights[0] = filter_x
    weights[1] = filter_y
    weights[2] = filter_diag1
    weights[3] = filter_diag2
    weights = weights.to(flow.device)

    # split the flow into flow_u and flow_v, conv them with the weights
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask

def create_mask(tensor, paddings):
    """
    tensor shape: [b, c, h, w]
    paddings: [2 x 2] shape list, the first row indicates up and down paddings
    the second row indicates left and right paddings
    |            |
    |       x    |
    |     x * x  |
    |       x    |
    |            |
    """
    shape = tensor.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])
    torch_paddings = [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]]  # left, right, up and down
    mask2d = F.pad(inner, pad=torch_paddings)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()

def ternary_loss(flow_comp, flow_gt, mask, current_frame, shift_frame, scale_factor=1):
    if scale_factor != 1:
        current_frame = F.interpolate(current_frame, scale_factor=1 / scale_factor, mode='bilinear')
        shift_frame = F.interpolate(shift_frame, scale_factor=1 / scale_factor, mode='bilinear')
    warped_sc = flow_warp(shift_frame, flow_gt.permute(0, 2, 3, 1))
    noc_mask = torch.exp(-50. * torch.sum(torch.abs(current_frame - warped_sc), dim=1).pow(2)).unsqueeze(1)
    warped_comp_sc = flow_warp(shift_frame, flow_comp.permute(0, 2, 3, 1))
    loss = ternary_loss2(current_frame, warped_comp_sc, noc_mask, mask)
    return loss

class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows, masks, frames):
        # pred_flows: b t-1 2 h w
        loss = 0
        warp_loss = 0
        h, w = pred_flows[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        frames0 = frames[:,:-1,...]
        frames1 = frames[:,1:,...]
        current_frames = [frames0, frames1]
        next_frames = [frames1, frames0]
        for i in range(len(pred_flows)):
            # print(pred_flows[i].shape)
            combined_flow = pred_flows[i] * masks[i] + gt_flows[i] * (1-masks[i])
            l1_loss = self.l1_criterion(pred_flows[i] * masks[i], gt_flows[i] * masks[i]) / torch.mean(masks[i])
            l1_loss += self.l1_criterion(pred_flows[i] * (1-masks[i]), gt_flows[i] * (1-masks[i])) / torch.mean((1-masks[i]))

            smooth_loss = smoothness_loss(combined_flow.reshape(-1,2,h,w), masks[i].reshape(-1,1,h,w))
            smooth_loss2 = second_order_loss(combined_flow.reshape(-1,2,h,w), masks[i].reshape(-1,1,h,w))
            
            warp_loss_i = ternary_loss(combined_flow.reshape(-1,2,h,w), gt_flows[i].reshape(-1,2,h,w), 
                            masks[i].reshape(-1,1,h,w), current_frames[i].reshape(-1,3,h,w), next_frames[i].reshape(-1,3,h,w)) 

            loss += l1_loss + smooth_loss + smooth_loss2

            warp_loss += warp_loss_i
            
        return loss, warp_loss


def edgeLoss(preds_edges, edges):
    """

    Args:
        preds_edges: with shape [b, c, h , w]
        edges: with shape [b, c, h, w]

    Returns: Edge losses

    """
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float() # Shape: [b,].
    num_neg = c * h * w - num_pos # Shape: [b,].
    neg_weights = (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    pos_weights = (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    weight = neg_weights * mask + pos_weights * (1 - mask)  # weight for debug
    losses = F.binary_cross_entropy_with_logits(preds_edges.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.mean(losses)
    return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_edges, gt_edges, masks):
        # pred_flows: b t-1 1 h w
        loss = 0
        h, w = pred_edges[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        for i in range(len(pred_edges)):
            # print(f'edges_{i}',  torch.sum(gt_edges[i])) # debug
            combined_edge = pred_edges[i] * masks[i] + gt_edges[i] * (1-masks[i])
            edge_loss = (edgeLoss(pred_edges[i].reshape(-1,1,h,w), gt_edges[i].reshape(-1,1,h,w)) \
                        + 5 * edgeLoss(combined_edge.reshape(-1,1,h,w), gt_edges[i].reshape(-1,1,h,w)))
            loss += edge_loss 

        return loss


class FlowSimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows):
        # pred_flows: b t-1 2 h w
        loss = 0
        h, w = pred_flows[0].shape[-2:]
        h_orig, w_orig = gt_flows[0].shape[-2:]
        pred_flows = [f.view(-1, 2, h, w) for f in pred_flows]
        gt_flows = [f.view(-1, 2, h_orig, w_orig) for f in gt_flows]

        ds_factor = 1.0*h/h_orig
        gt_flows = [F.interpolate(f, scale_factor=ds_factor, mode='area') * ds_factor for f in gt_flows]
        for i in range(len(pred_flows)):
            loss += self.l1_criterion(pred_flows[i], gt_flows[i])

        return loss
