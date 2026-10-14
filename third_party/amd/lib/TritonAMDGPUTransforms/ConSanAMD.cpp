#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"

namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;
namespace tti = mlir::triton::instrument;

using tti::BarrierInitInfo;
using tti::BarrierInvalidateInfo;
using tti::BarrierWaitInfo;
using tti::CommitKindDesc;
using tti::MemEffectsOpInfo;
using tti::WaitOpInfo;

namespace mlir {

class AMDConSanHooks : public tti::ConSanTargetHooks {
public:
  bool isTMAOp(Operation *op) const override {
    return isa<ttag::TDMOpInterface, ttag::AsyncTDMFusedCopyGlobalToLocalOp>(
        op);
  }

  // TDM ops from the same warp complete in issue order. ConSan's thread model
  // uses one logical TDM thread per WS partition, so the outstanding-commit
  // check excludes the calling thread's own column to avoid intra-partition
  // false positives while still detecting cross-partition races.
  bool isOrderedCommitKind(tti::CommitKind::Kind kind) const override {
    return kind == tti::CommitKind::TmaStore;
  }

  std::optional<BarrierInitInfo>
  getBarrierInitInfo(Operation *op) const override {
    if (auto initOp = dyn_cast<ttag::InitBarrierOp>(op))
      return BarrierInitInfo{initOp.getBarrier(), initOp.getCount()};
    return std::nullopt;
  }

  std::optional<BarrierWaitInfo>
  getBarrierWaitInfo(Operation *op) const override {
    if (auto waitOp = dyn_cast<ttag::WaitBarrierOp>(op))
      return BarrierWaitInfo{waitOp.getBarrier(), waitOp.getPhase(),
                             /*pred=*/Value()};
    return std::nullopt;
  }

  std::optional<BarrierInvalidateInfo>
  getBarrierInvalidateInfo(Operation *op) const override {
    return std::nullopt;
  }

  bool barrierWritesInvalidate() const override { return true; }

  std::optional<WaitOpInfo>
  getWaitOpInfo(Operation *op, const tti::AuxDataMap &auxData) const override {
    // On asyncmark targets (CDNA3/CDNA4), ttg::AsyncWaitOp is kept as-is
    // by UpdateAsyncWaitCount — read the commit group count directly.
    if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
      return WaitOpInfo{tti::CommitKind::AsyncCp, (int)asyncWaitOp.getNum(),
                        /*transferWrites=*/true, auxData.hasAsyncCopyReads};
    }
    // On non-asyncmark targets, amdgpu::AsyncWaitOp replaces ttg::AsyncWaitOp
    // after UpdateAsyncWaitCount. Read the preserved commit-group count.
    if (auto asyncWaitOp = dyn_cast<ttag::AsyncWaitOp>(op)) {
      if (auto attr = asyncWaitOp->getAttrOfType<IntegerAttr>(
              "ttg.num_commit_groups")) {
        return WaitOpInfo{tti::CommitKind::AsyncCp, (int)attr.getInt(),
                          /*transferWrites=*/true, auxData.hasAsyncCopyReads};
      }
      return std::nullopt;
    }
    if (auto tdmWaitOp = dyn_cast<ttag::AsyncTDMWait>(op)) {
      return WaitOpInfo{tti::CommitKind::TmaStore,
                        static_cast<int>(tdmWaitOp.getNum()),
                        /*transferWrites=*/true, /*transferReads=*/true};
    }
    // AMD AsyncTDMIntrinsicWait: replaces AsyncTDMWait after
    // UpdateAsyncWaitCount. Read the preserved TDM operation count.
    if (auto tdmWaitOp = dyn_cast<ttag::AsyncTDMIntrinsicWait>(op)) {
      if (auto attr =
              tdmWaitOp->getAttrOfType<IntegerAttr>("ttg.num_tdm_ops")) {
        return WaitOpInfo{tti::CommitKind::TmaStore, (int)attr.getInt(),
                          /*transferWrites=*/true, /*transferReads=*/true};
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  Value getIssuerCTAPred(ImplicitLocOpBuilder & /*b*/,
                         Operation * /*op*/) const override {
    return nullptr;
  }

  SmallVector<Operation *>
  createInitClusterBarrier(ImplicitLocOpBuilder &b) const override {
    auto arrive = ttag::ClusterBarrierArriveOp::create(b, b.getLoc());
    auto wait = ttag::ClusterBarrierWaitOp::create(b, b.getLoc());
    return {arrive.getOperation(), wait.getOperation()};
  }

  bool hasUnsummarizableCalleeState(Operation *op) const override {
    return isa<ttag::ClusterBarrierArriveOp, ttag::ClusterBarrierWaitOp>(op);
  }

  std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const override {
    bool isAsyncCopy =
        isa<ttag::BufferLoadToLocalOp, ttag::AsyncCopyLocalToGlobalOp>(op);
    if (isAsyncCopy || isTMAOp(op)) {
      MemEffectsOpInfo info = *ConSanTargetHooks::getMemEffectsOpInfo(op);
      Value barrier;
      if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op))
        barrier = barrierOp.getBarrier();
      if (barrier) {
        // mbarrier counts are block-level: a TDM op with a barrier is one
        // arrival, however many warps the lowering keeps active.
        // TODO: pass the descriptor's pred so a predicated-off op doesn't
        // count.
        info.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info.barriers.push_back({barrier, nullptr, /*count=*/1});
      } else {
        info.trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
        info.commitKind =
            isAsyncCopy ? tti::CommitKind::AsyncCp : tti::CommitKind::TmaStore;
        info.implicitCommit = !isAsyncCopy;
      }
      return info;
    }
    // AMD ArriveBarrierOp: explicit block-level arrival of `count` units.
    if (auto arriveOp = dyn_cast<ttag::ArriveBarrierOp>(op)) {
      MemEffectsOpInfo info;
      info.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info.barriers.push_back(
          {arriveOp.getBarrier(), nullptr, (int)arriveOp.getCount()});
      return info;
    }

    return ConSanTargetHooks::getMemEffectsOpInfo(op);
  }

  SmallVector<CommitKindDesc> getOutstandingWriteCommitKinds() const override {
    return {{tti::CommitKind::AsyncCp, "async_copy_global_to_shared"},
            {tti::CommitKind::TmaStore, "async_tdm_global_to_shared"}};
  }

  SmallVector<CommitKindDesc>
  getOutstandingReadCommitKinds(const tti::AuxDataMap &auxData) const override {
    if (!auxData.hasAsyncCopyReads)
      return {{tti::CommitKind::TmaStore, "async_tdm_shared_to_global"}};
    return {{tti::CommitKind::AsyncCp, "async_copy_shared_to_global"},
            {tti::CommitKind::TmaStore, "async_tdm_shared_to_global"}};
  }

  SmallVector<tti::CommitKind::Kind>
  getRequiredCommitKinds(ModuleOp module) const override {
    SmallVector<tti::CommitKind::Kind> kinds;
    bool needsTdm = false;
    bool needsAsyncCp = false;
    module.walk([&](Operation *op) {
      needsTdm |= isTMAOp(op) ||
                  isa<ttag::AsyncTDMWait, ttag::AsyncTDMIntrinsicWait>(op);
      needsAsyncCp |=
          isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp,
              ttg::AsyncWaitOp, ttag::BufferLoadToLocalOp,
              ttag::AsyncCopyLocalToGlobalOp, ttag::AsyncWaitOp>(op);
    });
    if (needsTdm)
      kinds.push_back(tti::CommitKind::TmaStore);
    if (needsAsyncCp)
      kinds.push_back(tti::CommitKind::AsyncCp);
    return kinds;
  }
};

void registerConSanAMDHooks() {
  tti::registerConSanHooks("amd",
                           [] { return std::make_unique<AMDConSanHooks>(); });
}

} // namespace mlir
