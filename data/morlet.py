import jax
import jax.numpy as jnp
from functools import partial   # 함수의 일부 인자를 고정하여 새로운 함수를 만들 때 사용됨


def get_bc(cycles, freq, k_sd=5):
    """
    주어진 cycles와 freq에 따라 wavelet의 폭(표준편차, bandwidth)을 계산하는 함수
    cycles 수가 많을수록, freq가 낮을수록 wavelet 폭이 넓어짐
    """
    return cycles/(k_sd * freq)


def cxmorelet(freq, cycles, sampling_freq):
    """ 중심 주파수와 사이클 수에 따라 복소 Morlet wavelet을 생성하는 함수 """
    # -1초~1초까지 총 2 * sampling_freq개의 샘플 생성 (2초간격의 시간축)
    t = jnp.linspace(-1, 1, sampling_freq*2)

    bc = get_bc(cycles, freq)           # wavelet 폭(표준편차) 계산
    norm = 1/(bc * jnp.sqrt(2*jnp.pi))  # Gaussian 정규화 계수 (Gaussian 함수의 면적이 1이 되도록)
    gauss = jnp.exp(-t**2/(2*bc**2))    # 시간 도메인의 Gaussian 파형 (시간축 기준으로 종 모양 곡선을 가지는 함수로, 중심을 기준으로 좌우 대칭)
    sine = jnp.exp(1j*2*jnp.pi*freq*t)  # 중심 주파수를 가진 복소수 sine 파형 (Euler 공식 사용) e^i2πft = cos(2πft) + isin(2πft)

    # Morlet wavelet = Gaussian × 복소 sine 파형 → 시간에 따라 감소되는 주기성 파형
    wavelet = norm * gauss * sine

    # 전체 파형의 절댓값 총합으로 정규화하여 에너지 보존
    return wavelet / jnp.sum(jnp.abs(wavelet))


@partial(jax.jit, static_argnums=3)                 # JIT 컴파일로 속도 향상
@partial(jax.vmap, in_axes=(None, 0, None, None))   # jax.vmap: 여러 주파수(freq)에 대해 병렬 연산 가능하게 벡터화
def wavelet_transform(signal, freq, cycles, sampling_freq):
    """ 입력된 signal에 대해 주파수별로 Morlet wavelet 변환 수행 """
    # 해당 주파수의 wavelet 생성
    wavelet = cxmorelet(freq, cycles, sampling_freq)

    # 생성된 wavelet과 signal 간 컨볼루션 수행
    # "same" 모드는 입력과 동일한 길이로 출력 유지
    return jax.scipy.signal.convolve(signal, wavelet, mode="same")