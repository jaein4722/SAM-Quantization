import argparse
import os
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn


def _patch_image_encoder_forward_for_global(image_encoder: nn.Module) -> None:
    """
    SlimSAM 학습 체크포인트(torch.load로 로드되는 글로벌 변형)를 사용할 때,
    `image_encoder.forward`가 최종 feature map만 반환하도록 inference.py와 동일하게 패치한다.
    """
    import types

    def forward(self, x):
        x = self.patch_embed(x)
        if getattr(self, "pos_embed", None) is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x, qkv_emb, mid_emb, x_emb = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

    image_encoder.forward = types.MethodType(forward, image_encoder)


def _load_sam_model(
    checkpoint: Optional[str],
    model_type: Optional[str],
    use_torch_load: bool,
    device: str,
    patch_forward: bool,
):
    """
    SlimSAM/SAM 모델 로드 유틸.

    - use_torch_load=True: torch.save로 저장된 전체 모델을 로드(예: SlimSAM 글로벌 학습 가중치)
    - use_torch_load=False: sam_model_registry에서 생성 후 checkpoint 로드
    - patch_forward=True: inference.py와 동일하게 image_encoder.forward 패치
    """
    if use_torch_load:
        assert checkpoint is not None and os.path.isfile(checkpoint), "--checkpoint 경로가 필요합니다."
        model = torch.load(checkpoint, map_location=device, weights_only=False)
        # DDP 래핑 제거
        if hasattr(model.image_encoder, "module"):
            model.image_encoder = model.image_encoder.module
        if patch_forward:
            _patch_image_encoder_forward_for_global(model.image_encoder)
        model.to(device)
        model.eval()
        return model

    # sam_model_registry 경유 로드
    from segment_anything import sam_model_registry

    assert model_type is not None, "--model-type 이 필요합니다. (예: vit_b, vit_h, vit_p77 등)"
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


def quantize_slimsam_dynamic(
    model: nn.Module,
    quantize_image_encoder: bool = True,
    quantize_prompt_encoder: bool = True,
    quantize_mask_decoder: bool = True,
) -> nn.Module:
    """
    PyTorch dynamic PTQ로 Linear 계층을 INT8로 양자화한다. (CPU 전용)
    - Transformer 기반 구조에서 Linear 비중이 크므로 효과적이며, 별도 캘리브레이션이 필요 없다.
    - Conv/LayerNorm/GELU 등은 dynamic quant 대상이 아니다.
    """
    # dynamic quantization은 CPU 백엔드에서 동작. 우선 CPU로 이동.
    model = model.to("cpu")
    model.eval()

    def _dq(m: nn.Module) -> nn.Module:
        if m is None:
            return None
        return torch.quantization.quantize_dynamic(
            m,
            {nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        )

    if quantize_image_encoder and hasattr(model, "image_encoder"):
        model.image_encoder = _dq(model.image_encoder)
    if quantize_prompt_encoder and hasattr(model, "prompt_encoder"):
        model.prompt_encoder = _dq(model.prompt_encoder)
    if quantize_mask_decoder and hasattr(model, "mask_decoder"):
        model.mask_decoder = _dq(model.mask_decoder)

    return model


@torch.no_grad()
def _fake_quantize_weight_per_channel(w: torch.Tensor) -> torch.Tensor:
    """대칭 per-channel 가중치 fake-quantize (int8→dequantize). 채널은 dim=0으로 가정.
    Linear: [out_features, in_features]
    Conv2d: [out_channels, in_channels, kH, kW]
    """
    orig_dtype = w.dtype
    w_view = w.detach()
    c = w_view.shape[0]
    w_flat = w_view.reshape(c, -1)
    max_abs = w_flat.abs().amax(dim=1)  # [C]
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))  # [C]
    # q = clamp(round(w/scale), -127, 127)
    # reshape scale for broadcasting
    new_shape = [c] + [1] * (w_view.dim() - 1)
    scale_bc = scale.reshape(new_shape)
    q = torch.round(w_view / scale_bc).clamp_(-127, 127).to(torch.int8)
    dq = q.to(torch.float32) * scale_bc
    return dq.to(orig_dtype)


@torch.no_grad()
def _fake_quantize_weight_per_tensor(w: torch.Tensor) -> torch.Tensor:
    """대칭 per-tensor 가중치 fake-quantize (int8→dequantize)."""
    orig_dtype = w.dtype
    max_abs = w.detach().abs().max()
    scale = (max_abs / 127.0) if max_abs > 0 else torch.tensor(1.0, device=w.device, dtype=torch.float32)
    q = torch.round(w.detach() / scale).clamp_(-127, 127).to(torch.int8)
    dq = q.to(torch.float32) * scale
    return dq.to(orig_dtype)


def _apply_minmax_fake_quant(
    module: nn.Module,
    include_conv: bool,
    per_tensor: bool,
) -> Tuple[int, float, float]:
    """모듈 트리 전체를 순회하며 Linear(및 선택적으로 Conv2d) 가중치에 MinMax fake-quant 적용.
    반환: (수정한 모듈 수, 수정된 가중치의 최대 차이 합, 평균 차이 합)
    """
    changed = 0
    sum_max_diff = 0.0
    sum_mean_diff = 0.0
    target_types = (nn.Linear,) + ((nn.Conv2d,) if include_conv else tuple())

    for m in module.modules():
        if isinstance(m, target_types) and hasattr(m, "weight"):
            with torch.no_grad():
                w0 = m.weight.detach().clone()
                if per_tensor:
                    dq = _fake_quantize_weight_per_tensor(w0)
                else:
                    dq = _fake_quantize_weight_per_channel(w0)
                m.weight.data.copy_(dq)
                d = (dq - w0).abs()
                sum_max_diff += float(d.max().item()) if d.numel() > 0 else 0.0
                sum_mean_diff += float(d.mean().item()) if d.numel() > 0 else 0.0
                changed += 1
    return changed, sum_max_diff, sum_mean_diff


def quantize_slimsam_minmax_weights(
    model: nn.Module,
    quantize_image_encoder: bool = True,
    quantize_prompt_encoder: bool = True,
    quantize_mask_decoder: bool = True,
    include_conv: bool = False,
    per_tensor: bool = False,
) -> nn.Module:
    """
    MinMax 기반 weight-only fake quantization.
    - Linear(기본) 가중치에 대하여 대칭 INT8 per-channel(기본) 또는 per-tensor 스케일로 quant→dequant 적용.
    - include_conv=True 시 Conv2d 가중치에도 동일 적용.
    - 실행 시 모델 dtype은 그대로 유지되어 CPU/GPU 모두에서 동작.
    """
    model.eval()
    total_changed = 0
    total_max_diff = 0.0
    total_mean_diff = 0.0
    if quantize_image_encoder and hasattr(model, "image_encoder"):
        c, mx, mn = _apply_minmax_fake_quant(model.image_encoder, include_conv=include_conv, per_tensor=per_tensor)
        total_changed += c; total_max_diff += mx; total_mean_diff += mn
    if quantize_prompt_encoder and hasattr(model, "prompt_encoder"):
        c, mx, mn = _apply_minmax_fake_quant(model.prompt_encoder, include_conv=include_conv, per_tensor=per_tensor)
        total_changed += c; total_max_diff += mx; total_mean_diff += mn
    if quantize_mask_decoder and hasattr(model, "mask_decoder"):
        c, mx, mn = _apply_minmax_fake_quant(model.mask_decoder, include_conv=include_conv, per_tensor=per_tensor)
        total_changed += c; total_max_diff += mx; total_mean_diff += mn
    # 요약 저장을 위해 속성에 기록 (선택적)
    setattr(model, "_minmax_changed_layers", int(total_changed))
    setattr(model, "_minmax_sum_max_diff", float(total_max_diff))
    setattr(model, "_minmax_sum_mean_diff", float(total_mean_diff))
    return model


def save_quantized_model(model: nn.Module, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model, output_path)

# =========================
# Static PTQ (activation fake-quant with calibration)
# =========================

class MinMaxObserver:
    def __init__(self):
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        x_detached = x.detach()
        cur_min = x_detached.amin()
        cur_max = x_detached.amax()
        if self.min_val is None:
            self.min_val = cur_min
            self.max_val = cur_max
        else:
            self.min_val = torch.minimum(self.min_val, cur_min)
            self.max_val = torch.maximum(self.max_val, cur_max)

    @torch.no_grad()
    def get_scale(self) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            return torch.tensor(1.0)
        max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs())
        scale = max_abs / 127.0
        scale = torch.where(scale > 0, scale, torch.tensor(1.0, device=scale.device, dtype=scale.dtype))
        return scale


class FakeQuantize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observer = MinMaxObserver()
        self.calibrate_mode: bool = False
        self.enabled: bool = False

    def set_calibrate(self, is_calibrate: bool) -> None:
        self.calibrate_mode = is_calibrate

    def set_enable(self, is_enabled: bool) -> None:
        self.enabled = is_enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrate_mode:
            self.observer.observe(x)
            return x
        if self.enabled:
            scale = self.observer.get_scale()
            q = torch.round(x / scale).clamp_(-127, 127).to(torch.int8)
            x = q.to(torch.float32) * scale
            return x.to(dtype=x.dtype)
        return x


class StaticQuantWrapper(nn.Module):
    def __init__(self, module: nn.Module, quantize_input: bool = True, quantize_output: bool = True) -> None:
        super().__init__()
        self.module = module
        self.input_quant = FakeQuantize() if quantize_input else None
        self.output_quant = FakeQuantize() if quantize_output else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quant is not None:
            x = self.input_quant(x)
        x = self.module(x)
        if self.output_quant is not None:
            x = self.output_quant(x)
        return x


def _insert_static_act_quant(
    module: nn.Module,
    include_conv: bool,
    quantize_input: bool,
    quantize_output: bool,
) -> int:
    """nn.Linear (옵션으로 nn.Conv2d)을 StaticQuantWrapper로 교체. 반환: 래핑된 모듈 수"""
    count = 0
    target_types = (nn.Linear,) + ((nn.Conv2d,) if include_conv else tuple())

    for name, child in list(module.named_children()):
        if isinstance(child, StaticQuantWrapper):
            continue
        if isinstance(child, target_types):
            setattr(module, name, StaticQuantWrapper(child, quantize_input=quantize_input, quantize_output=quantize_output))
            count += 1
        else:
            count += _insert_static_act_quant(child, include_conv, quantize_input, quantize_output)
    return count


def _set_calibration_mode(module: nn.Module, calibrate: bool) -> None:
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.set_calibrate(calibrate)
            if calibrate:
                m.set_enable(False)


def _set_quantize_enabled(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.set_enable(enabled)
            if enabled:
                m.set_calibrate(False)


def calibrate_with_images(
    model: nn.Module,
    images_dir: str,
    max_images: int,
    device: str = "cpu",
    run_decoder: bool = True,
) -> None:
    """이미지 폴더를 사용해 activation MinMax 캘리브레이션 수행."""
    import os
    import glob
    import numpy as np
    import cv2
    from segment_anything import SamPredictor

    model.to(device).eval()
    predictor = SamPredictor(model)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(images_dir, e)))
    files = sorted(files)[: max_images]

    _set_calibration_mode(model, True)
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        if run_decoder:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            pts = np.array([[cx, cy]], dtype=np.float32)
            lbs = np.array([1], dtype=np.int32)
            with torch.no_grad():
                predictor.predict(point_coords=pts, point_labels=lbs, multimask_output=False)
    _set_calibration_mode(model, False)


# =========================
# Q-SAM2 스타일 선형층 보정 (릿지 회귀)
# =========================
@torch.no_grad()
def _flatten_to_2d(x: torch.Tensor) -> torch.Tensor:
    """
    (N, *, C) 꼴 텐서를 (N_total, C)로 평탄화.
    Linear 입력/출력에 공통 적용하여 X:[K, din], Y:[K, dout] 형태로 만든다.
    """
    x = x.detach()
    if x.dim() == 2:
        return x
    # 마지막 채널을 feature로 간주
    c = x.shape[-1]
    return x.reshape(-1, c)

def collect_linear_xy_with_hooks(model: nn.Module, image_paths: List[str], max_images: int = 64, device: str = "cpu") -> Dict[str, tuple]:
    """
    SamPredictor 경로를 통해 한 번씩 추론을 흘리면서 각 Linear 층의 (X, Y_b) 수집.
    - X: 선형층 입력, 2D (K, din)
    - Y_b: 선형층 출력에서 bias 제거, 2D (K, dout)
    """
    import numpy as np
    import cv2
    from segment_anything import SamPredictor

    model.eval().to(device)
    inputs_cache: Dict[str, List[torch.Tensor]] = {}
    outputs_cache: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def pre_hook(name):
        def _pre(mod, inp):
            if len(inp) == 0:
                return
            x = inp[0]
            x2d = _flatten_to_2d(x).to("cpu").float()
            inputs_cache.setdefault(name, []).append(x2d)
        return _pre

    def post_hook(name):
        def _post(mod, _, out):
            y = out
            # bias 제거
            if getattr(mod, "bias", None) is not None:
                # out shape: (..., dout). broadcast-safe 빼기 위해 bias reshape
                b = mod.bias.detach().view(1, -1)
                y = y - b
            y2d = _flatten_to_2d(y).to("cpu").float()
            outputs_cache.setdefault(name, []).append(y2d)
        return _post

    named_linears = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            named_linears.append((n, m))
            handles += [
                m.register_forward_pre_hook(pre_hook(n)),
                m.register_forward_hook(post_hook(n)),
            ]

    predictor = SamPredictor(model)

    used = 0
    for p in image_paths:
        if used >= max_images:
            break
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        pts = np.array([[cx, cy]], dtype=np.float32)
        lbs = np.array([1], dtype=np.int32)
        # 한 번만 예측을 흘려도 모든 hook이 작동
        _ = predictor.predict(point_coords=pts, point_labels=lbs, box=None, multimask_output=False)
        used += 1

    for h in handles:
        h.remove()

    XY = {}
    for n, _ in named_linears:
        if n in inputs_cache and n in outputs_cache:
            X = torch.cat(inputs_cache[n], dim=0)
            Y = torch.cat(outputs_cache[n], dim=0)
            # din/dout 일치하는 페어만 사용
            if X.numel() > 0 and Y.numel() > 0 and X.shape[0] == Y.shape[0]:
                XY[n] = (X, Y)
    return XY

@torch.no_grad()
def calibrate_linear_weights_ridge(model: nn.Module, XY: Dict[str, tuple], lambda_reg: float = 1e-3) -> nn.Module:
    """
    리지(틱호노프) 해로 W, b 를 재추정.
    y = x W^T + b  (Linear 규약)
    W^T = (X^T X + λI)^{-1} X^T Y
    b   = mean(Y - X W^T)
    """
    eye_cache: Dict[int, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in XY:
            X, Y = XY[name]  # X:[K,din], Y:[K,dout]
            X = X.float()
            Y = Y.float()
            din = X.shape[1]
            # (X^T X + λI)
            if din not in eye_cache:
                eye_cache[din] = torch.eye(din, dtype=torch.float32, device=X.device) # Ensure eye is on correct device
            A = X.T @ X + lambda_reg * eye_cache[din]
            B = X.T @ Y  # [din, dout]
            W_t = torch.linalg.solve(A, B)  # [din, dout]
            W_new = W_t.T.contiguous()      # [dout, din]
            mod.weight.data.copy_(W_new.to(mod.weight.dtype))
            # bias 재설정
            b_new = (Y - X @ W_t).mean(dim=0)
            if mod.bias is not None:
                mod.bias.data.copy_(b_new.to(mod.bias.dtype))
    return model

def run_linear_calibration_pass(
    model: nn.Module,
    calib_dir: str,
    calib_size: int = 64,
    lambda_reg: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """
    calib_dir의 이미지들로 SamPredictor를 통해 한 번씩 forward하여 (X,Y) 수집 → 리지 보정 → 가중치 치환.
    """
    import pathlib
    img_paths = []
    p = pathlib.Path(calib_dir)
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
        img_paths += list(p.rglob(ext))
    img_paths = [str(x) for x in sorted(img_paths)]
    if not img_paths:
        raise FileNotFoundError(f"캘리브레이션용 이미지가 없습니다: {calib_dir}")
    XY = collect_linear_xy_with_hooks(model, img_paths, max_images=calib_size, device=device)
    model = calibrate_linear_weights_ridge(model, XY, lambda_reg=lambda_reg)
    return model


def run_single_verify(
    model: nn.Module,
    image_path: str,
    point: Optional[Tuple[int, int]] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    multimask_output: bool = False,
):
    """
    간단 검증: 이미지 1장을 CPU에서 로드하고 SamPredictor로 한 번 추론해 본다.
    - dynamic PTQ 결과는 CPU 전용이므로 device는 항상 CPU로 강제한다.
    - 예시로 point 또는 box 중 하나가 주어지면 해당 프롬프트로 추론한다.
    """
    import numpy as np
    import cv2
    from segment_anything import SamPredictor

    device = torch.device("cpu")
    model.to(device).eval()
    predictor = SamPredictor(model)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    with torch.no_grad():
        if point is not None:
            pts = np.array([list(point)], dtype=np.float32)
            lbs = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=lbs,
                box=None,
                multimask_output=multimask_output,
            )
        elif box is not None:
            bx = np.array(list(box), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bx,
                multimask_output=multimask_output,
            )
        else:
            # 기본: 중앙 한 점을 양성 포인트로 사용
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            pts = np.array([[cx, cy]], dtype=np.float32)
            lbs = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=lbs,
                box=None,
                multimask_output=multimask_output,
            )

    return masks, scores


def main():
    parser = argparse.ArgumentParser(description="SlimSAM PTQ: dynamic, MinMax(weight-only), static(activation)")
    group_load = parser.add_argument_group("load")
    group_load.add_argument("--checkpoint", type=str, default=None, help="모델 체크포인트 경로")
    group_load.add_argument("--model-type", type=str, default=None, help="sam_model_registry 키 (예: vit_b, vit_h, vit_p77)")
    group_load.add_argument("--use-torch-load", action="store_true", help="torch.load로 전체 모델 로드")
    group_load.add_argument("--patch-forward", action="store_true", help="inference.py와 같이 image_encoder.forward 패치")

    group_q = parser.add_argument_group("quant")
    group_q.add_argument("--method", type=str, choices=["dynamic", "minmax", "static"], default="dynamic", help="PTQ 방식 선택")
    group_q.add_argument("--no-enc", action="store_true", help="image_encoder 양자화 비활성화")
    group_q.add_argument("--no-prompt", action="store_true", help="prompt_encoder 양자화 비활성화")
    group_q.add_argument("--no-dec", action="store_true", help="mask_decoder 양자화 비활성화")
    # MinMax 관련 옵션
    group_q.add_argument("--minmax-per-tensor", action="store_true", help="MinMax를 per-tensor로 적용 (기본: per-channel)")
    group_q.add_argument("--include-conv", action="store_true", help="MinMax에서 Conv2d 가중치도 포함")
    group_q.add_argument("--minmax-verbose", action="store_true", help="MinMax 적용 요약(변경 레이어 수/차이) 출력")

    # Static PTQ (activation) 옵션
    group_static = parser.add_argument_group("static-ptq")
    group_static.add_argument("--static-include-conv", action="store_true", help="Static에서 Conv2d 활성도 래핑")
    group_static.add_argument("--static-quant-input", action="store_true", help="각 레이어 입력 활성 quant", default=True)
    group_static.add_argument("--static-quant-output", action="store_true", help="각 레이어 출력 활성 quant", default=True)
    group_static.add_argument("--static-run-decoder", action="store_true", help="캘리브레이션 시 mask decoder까지 실행", default=False)

    # Calibration (Q-SAM2 style) 옵션
    group_cal = parser.add_argument_group("calibration")
    group_cal.add_argument("--do-calib", action="store_true", help="선형층 리지 보정 수행 후 MinMax 적용")
    group_cal.add_argument("--calib-dir", type=str, default=None, help="캘리브레이션용 이미지 디렉터리")
    group_cal.add_argument("--calib-size", type=int, default=64, help="캘리브 이미지 개수 상한")
    group_cal.add_argument("--lambda-reg", type=float, default=1e-3, help="릿지 정규화 계수 λ")

    group_out = parser.add_argument_group("output")
    group_out.add_argument("--output", type=str, required=True, help="양자화된 모델 저장 경로 (*.pth)")

    group_verify = parser.add_argument_group("verify")
    group_verify.add_argument("--verify-image", type=str, default=None, help="간단 검증용 이미지 경로")
    group_verify.add_argument("--verify-point", type=int, nargs=2, default=None, metavar=("X", "Y"), help="검증용 포인트 (x y)")
    group_verify.add_argument("--verify-box", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"), help="검증용 박스 (x0 y0 x1 y1)")

    args = parser.parse_args()

    # 모델 로드 (기본은 CPU로 로드하여 바로 양자화 가능하도록 함)
    device_for_load = "cpu" if args.use_torch_load else ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_sam_model(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        use_torch_load=args.use_torch_load,
        device=device_for_load,
        patch_forward=args.patch_forward,
    )

    # ====== (옵션) Q-SAM2 선형층 보정 ======
    # - weight-only PTQ 전 단계에서 가중치를 보정하여 양자화 민감도 완화
    if args.method == "minmax" and args.do_calib:
        if args.calib_dir is None:
            raise ValueError("--do-calib 사용 시 --calib-dir 가 필요합니다.")
        # 보정은 CPU 또는 가용 CUDA에서 수행 가능 (hook/수집은 CPU로 옮겨 저장)
        device_for_calib = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Calib] collecting (X,Y) with {args.calib_size} images on {device_for_calib} …")
        model = run_linear_calibration_pass(
            model,
            calib_dir=args.calib_dir,
            calib_size=args.calib_size,
            lambda_reg=args.lambda_reg,
            device=device_for_calib,
        )
        print("[Calib] linear ridge calibration done.")

    # PTQ 수행
    if args.method == "dynamic":
        # Dynamic PTQ (CPU 전용, Linear 대상)
        q_model = quantize_slimsam_dynamic(
            model,
            quantize_image_encoder=not args.no_enc,
            quantize_prompt_encoder=not args.no_prompt,
            quantize_mask_decoder=not args.no_dec,
        )
    elif args.method == "minmax":
        # MinMax weight-only fake quant (CPU/GPU 호환)
        q_model = quantize_slimsam_minmax_weights(
            model,
            quantize_image_encoder=not args.no_enc,
            quantize_prompt_encoder=not args.no_prompt,
            quantize_mask_decoder=not args.no_dec,
            include_conv=args.include_conv,
            per_tensor=args.minmax_per_tensor,
        )
        if args.minmax_verbose:
            print(
                f"[MinMax] changed_layers={getattr(q_model, '_minmax_changed_layers', -1)} "
                f"sum_max_diff={getattr(q_model, '_minmax_sum_max_diff', -1.0):.6f} "
                f"sum_mean_diff={getattr(q_model, '_minmax_sum_mean_diff', -1.0):.6f}"
            )
    else:
        # Static activation fake-quant: 래핑 → 캘리브레이션 → 활성화
        # 1) 대상 서브모듈 지정
        targets: List[Tuple[str, bool]] = [
            ("image_encoder", not args.no_enc),
            ("prompt_encoder", not args.no_prompt),
            ("mask_decoder", not args.no_dec),
        ]
        wrapped = 0
        for attr, enable in targets:
            if enable and hasattr(model, attr):
                wrapped += _insert_static_act_quant(
                    getattr(model, attr),
                    include_conv=args.static_include_conv,
                    quantize_input=args.static_quant_input,
                    quantize_output=args.static_quant_output,
                )
        if args.calib_dir is None:
            raise ValueError("--method static 사용 시 --calib-dir 가 필요합니다.")
        device_for_static = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Static] wrapped_layers={wrapped}. Calibrating on {device_for_static} …")
        calibrate_with_images(
            model,
            images_dir=args.calib_dir,
            max_images=args.calib_size,
            device=device_for_static,
            run_decoder=args.static_run_decoder,
        )
        _set_quantize_enabled(model, True)
        q_model = model

    # 저장
    save_quantized_model(q_model, args.output)

    # 옵션: 간단 검증
    if args.verify_image is not None:
        _, score = run_single_verify(
            q_model,
            image_path=args.verify_image,
            point=tuple(args.verify_point) if args.verify_point is not None else None,
            box=tuple(args.verify_box) if args.verify_box is not None else None,
            multimask_output=False,
        )
        print(f"Score: {score}")


if __name__ == "__main__":
    main()

