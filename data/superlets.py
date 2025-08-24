import jax
import jax.numpy as jnp
from functools import partial
from MAE.morlet import wavelet_transform

@partial(jax.jit, static_argnums=3)                 # 정적 인수(sampling_freq) 기준 JIT 컴파일로 속도 향상
@partial(jax.vmap, in_axes=(None, None, 0, None))   # 각 order마다 병렬 변환 수행
def superlet_transform_helper(signal, freqs, order, sampling_freq):
    """
    여러 주파수에 대해 병렬로 wavelet 변환을 수행하는 보조 함수
    morlet 기반 변환 수행 후 √2로 정규화함
    """
    return wavelet_transform(signal, freqs, order, sampling_freq) * jnp.sqrt(2)


def order_to_cycles(base_cycle, max_order, mode):
    """
    주어진 order 수만큼 cycle 수 배열 생성하는 함수
    주파수 해상도를 조절할 수 있도록 다양한 cycle 수를 설정함
    additive (선형 증가): [base, base+1, ..., base+max_order-1]
    multiplicative (기하급수 증가): [base×1, base×2, ..., base×max_order]
    """
    if mode == "add":
        return jnp.arange(0, max_order) + base_cycle
    elif mode == "mul":
        return jnp.arange(1, max_order+1) * base_cycle
    else: raise ValueError("mode should be one of \"mul\" or \"add\"")


def get_order(f, f_min: int, f_max: int, o_min: int, o_max: int):
    """
    주어진 주파수 f에 대해 min/max를 기준으로 적절한 order를 선형 보간하여 할당하는 함수
    입력 주파수 f가 낮을수록 낮은 order, 높을수록 높은 order를 배정함
    """
    return o_min + round((o_max - o_min) * (f - f_min) / (f_max - f_min))


@partial(jax.vmap, in_axes=(0, None))
def get_mask(order, max_order):
    """
    주파수별 order에 따라 마스킹 Boolean 배열 생성하는 함수
    즉, order보다 큰 cycle에 해당하는 응답만 제거하기 위한 마스크를 만드는 함수
    주파수별로 order 이하의 응답은 keep (False), 초과는 제거(True)로 설정함
    예: order가 3이면 [1 > 3, 2 > 3, 3 > 3, 4 > 3, 5 > 3, ...] → [False, False, False, True, True, ...]
    """
    return jnp.arange(1, max_order+1) > order


@jax.jit
def norm_geomean(X, root_pows, eps):
    """
    기하평균 계산하는 함수
    입력 X (order × T)를 log → sum → exp 방식으로 안정적인 geometric mean을 계산함
    eps는 log(0) 방지용
    """
    X = jnp.log(X + eps).sum(axis=0)

    return jnp.exp(X / jnp.array(root_pows).reshape(-1, 1))


# @jax.jit
def adaptive_superlet_transform(signal, freqs, sampling_freq: int, base_cycle: int, min_order: int, max_order: int, eps=1e-12, mode="mul"):
    """
    Superlet 핵심 함수. 제공된 신호에 대해 Adaptive Superlet 변환을 수행
    signal을 주파수별 adaptive wavelet으로 변환해 2D TF map 생성

    Args:
        signal (jnp.ndarray): 분석할 1차원 신호 (시간에 따른 값)
        freqs (jnp.ndarray): 변환을 수행할 주파수들의 1차원 정렬 배열
        sampling_freq (int): 신호의 샘플링 주파수

        base_cycle (int): order = 1일 때 사용할 wavelet의 기본 주기 수
        min_order (int): 특정 주파수에서 사용할 Superlet order의 최소값
        max_order (int): 특정 주파수에서 사용할 Superlet order의 최대값

        eps (float, optional): 기하평균 계산에서 수치적 안정성을 위한 작은 수 (기본값: 1e-12)
        mode (str, optional):  wavelet 주기를 계산할 방식. “add” 또는 “mul” 중 하나 (기본값: “mul”)

    Returns:
        jnp.ndarray: 계산된 scalogram (Frequency x Time)의 2D array
    """
    # additive 또는 multiplicative 방식으로 다양한 cycle 수 설정
    cycles = order_to_cycles(base_cycle, max_order, mode)

    # 주파수별 적절한 order 계산
    orders = get_order(freqs, min(freqs), max(freqs), min_order, max_order)

    # order보다 긴 cycle에 해당하는 응답(고차 order) 제거용 마스킹  생성
    mask = get_mask(orders, max_order)

    # 전체 wavelet 응답 계산
    out = superlet_transform_helper(signal, freqs, cycles, sampling_freq)   # shape: (max_order, n_freqs, T)

    # out = jax.ops.index_update(out, mask.T, 1)

    # 마스킹된 응답은 1로 설정 (log(1) = 0)
    out = out.at[mask.T].set(1)

    # 유효한 응답들에 대해 주파수마다 geometric mean 계산 → 최종 TF map 완성
    return norm_geomean(out, orders, eps)