import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["MBarrierLayout", "init", "wait", "arrive"]


class MBarrierLayout(SwizzledSharedLayout):
    """
    Layout for mbarrier synchronization.

    Args:
        cga_layout (List[List[int]]): CGA layout bases. Defaults to [].
    """

    def __init__(self, cga_layout=None):
        super().__init__(vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=cga_layout or [])


@builtin
def init(mbarrier, count, _semantic=None):
    """
    Initialize an mbarrier with a specified count.

    An mbarrier consists of an init count, a pending count and a phase.
    At initialization, the init count and pending count are initialized with the
    given ``count`` and the phase is initialized to 0.

    ``count`` is the number of block-level arrivals that complete a phase.
    Each source contributes the same amount however the compiler lowers it:

    * a :mod:`tdm` load, store, gather, or scatter with ``mbarrier=`` contributes
      1, or 0 if its predicate is false;
    * :func:`arrive` with ``count=c`` contributes ``c``;
    * :func:`async_copy.mbarrier_arrive` contributes 1.

    The compiler scales ``count`` by ``num_warps``, so ``count * num_warps``
    must be at most 65536. Every warp-specialized partition that arrives on the
    barrier must currently have ``num_warps`` warps.

    Each phase must complete before any arrival for the next phase: the
    hardware requires that phases never drift more than 1 apart.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to initialize.
        count (int): The number of block-level arrivals per phase. Must be
            between 1 and 65536.
    """
    count = _unwrap_if_constexpr(count)
    _semantic.builder.create_lds_barrier_init(mbarrier.handle, count)


@builtin
def wait(mbarrier, phase, _semantic=None):
    """
    Wait until the parity of the mbarrier's phase differs from ``phase``.

    This means that the given ``phase`` has completed.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to wait on.
        phase (int): The phase parity to compare against, 0 or 1. The hardware
            phase is a 3-bit counter and only its parity is compared, so any
            other value makes the wait return immediately.
    """
    phase = _semantic.to_tensor(phase)

    _semantic.builder.create_lds_barrier_wait(mbarrier.handle, phase.handle)


@builtin
def arrive(mbarrier, *, count=1, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    Contributes ``count`` block-level arrivals (see :func:`init`) to the
    pending arrival count of the ``mbarrier``.
    If the pending count reaches zero, the phase changes (is decremented in a
    wraparound manner) and the pending count is reloaded with the init count value.
    Arrivals beyond the pending count are dropped, not carried into the next
    phase. The arrive is ordered after this program's earlier shared-memory
    accesses, but not after its earlier TDM ops.
    Returns the mbarrier's phase parity (0 for even, 1 for odd) prior to the
    ``arrive`` operation.

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Number of block-level arrivals, between 1 and 65536.
            Defaults to 1.

    Returns:
        int: Phase parity of the mbarrier prior to the ``arrive`` operation.
    """
    count = _unwrap_if_constexpr(count)
    handle = _semantic.builder.create_lds_barrier_arrive(mbarrier.handle, count)
    return ttgl.tensor(handle, ttgl.int32)
