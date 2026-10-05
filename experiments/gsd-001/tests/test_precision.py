from run_sweep import cuda_bf16_capable


class _Cuda:
    def __init__(self, capability):
        self._capability = capability

    def get_device_capability(self, index=0):
        assert index == 0
        return self._capability


class _Torch:
    def __init__(self, capability):
        self.cuda = _Cuda(capability)


def test_t4_is_not_vllm_bf16_capable():
    assert cuda_bf16_capable(_Torch((7, 5))) is False


def test_ampere_is_vllm_bf16_capable():
    assert cuda_bf16_capable(_Torch((8, 0))) is True
