// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm --verify-diagnostics | FileCheck %s --enable-var-scope --check-prefix=GFX1250

// GFX1250: [[$MMRA_TAG:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"local">

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // count * num-warps - 1 = 2 * 4 - 1 = 7, in both the pending and init count fields.
    // GFX1250:   %[[INIT_VAL1:.+]] = llvm.mlir.constant(30064771079 : i64) : i64
    // GFX1250:   %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1250:   llvm.cond_br %{{.+}}, ^[[BB0:.+]], ^[[BB1:.+]]
    // GFX1250-NEXT: ^[[BB0]]:
    // GFX1250-NEXT:   llvm.store %[[INIT_VAL1]], %[[ALLOC_PTR]] : i64, !llvm.ptr<3>
    // GFX1250-NEXT:   llvm.br ^[[BB1]]
    // GFX1250-NEXT: ^[[BB1]]:
    // GFX1250-NEXT:   llvm.fence syncscope("workgroup") release {llvm.mmra = [[$MMRA_TAG]]}
    // GFX1250-NEXT:   rocdl.s.barrier
    // GFX1250-NEXT:   llvm.fence syncscope("workgroup") acquire {llvm.mmra = [[$MMRA_TAG]]}
    // GFX1250-NEXT:   llvm.return
    amdg.init_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // GFX1250-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %phase: i32) {
    // GFX1250: rocdl.s.sleep {{.*}}
    // GFX1250: llvm.load {{.*}} : !llvm.ptr<3> -> i64
    // GFX1250: llvm.icmp "ne" {{%arg1, %.*|%.*, %arg1}} : i32
    amdg.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // GFX1250-LABEL: arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // GFX1250: %[[UPDATE_VAL1:.+]] = llvm.mlir.constant(1 : i64) : i64
    // GFX1250: %[[ZERO:.+]] = llvm.mlir.zero : i64
    // GFX1250: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1250: %[[LANE:.+]] = llvm.urem
    // GFX1250: %[[IS_LANE0:.+]] = llvm.icmp "eq" %[[LANE]]
    // GFX1250: llvm.cond_br %[[IS_LANE0]], ^[[ARRIVE:.+]], ^[[END:.+]](%[[ZERO]] : i64)
    // GFX1250-NEXT: ^[[ARRIVE]]:
    // GFX1250-NEXT: %[[STATE:.+]] = rocdl.ds.atomic.barrier.arrive.rtn.b64 %[[ALLOC_PTR]], %[[UPDATE_VAL1]] : !llvm.ptr<3>, i64 -> i64
    // GFX1250-NEXT: llvm.br ^[[END]](%[[STATE]] : i64)
    // GFX1250-NEXT: ^[[END]](%[[PRIOR:.+]]: i64):
    // GFX1250: %[[PHASE:.+]] = llvm.and
    // GFX1250-NEXT: rocdl.readfirstlane %[[PHASE]] : i32
    %prior_phase = amdg.arrive_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
    tt.return
  }

  // GFX1250-LABEL: async_copy_mbarrier_arrive
  tt.func @async_copy_mbarrier_arrive(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // GFX1250: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1250: %[[LANE:.+]] = llvm.urem
    // GFX1250: %[[IS_LANE0:.+]] = llvm.icmp "eq" %[[LANE]]
    // GFX1250: llvm.cond_br %[[IS_LANE0]], ^[[ARRIVE:.+]], ^[[END:.+]]
    // GFX1250-NEXT: ^[[ARRIVE]]:
    // GFX1250-NEXT: rocdl.ds.atomic.async.barrier.arrive.b64 %[[ALLOC_PTR]] : !llvm.ptr<3>
    // GFX1250-NEXT: llvm.br ^[[END]]
    amdg.async_copy_mbarrier_arrive %alloc : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @init_barrier_count_too_large(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @+2 {{mbarrier count 8193 times 8 warps exceeds the hardware limit of 65536 arrivals}}
    // expected-error @+1 {{failed to legalize operation}}
    amdg.init_barrier %alloc, 8193 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // GFX1250-LABEL: mbarrier_arrivals_in_partition
  tt.func @mbarrier_arrivals_in_partition(%desc: !tt.tensordesc<64x64xf16, #shared>, %dst: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %alloc: !ttg.memdesc<1xi64, #bar, #smem, mutable>) {
    ttg.warp_specialize(%desc, %dst, %alloc) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%d: !tt.tensordesc<64x64xf16, #shared>, %s: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(4) {
      // GFX1250: llvm.amdgcn.tensor.load.to.lds
      // GFX1250: rocdl.ds.atomic.barrier.arrive.rtn.b64
      // GFX1250: rocdl.ds.atomic.async.barrier.arrive.b64
      %0 = amdg.async_tdm_copy_global_to_local %d into %s, barrier = %b : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %1 = amdg.arrive_barrier %b, 1 : !ttg.memdesc<1xi64, #bar, #smem, mutable> -> i32
      amdg.async_copy_mbarrier_arrive %b : !ttg.memdesc<1xi64, #bar, #smem, mutable>
      ttg.warp_return
    } : (!tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 6 : i32} {
  tt.func @arrive_barrier_partition_warp_mismatch(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    ttg.warp_specialize(%alloc) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(2) {
      // expected-error @+2 {{mbarrier arrivals require every arriving region to have ttg.num-warps (4) warps, but this region has 2}}
      // expected-error @+1 {{failed to legalize operation}}
      %0 = amdg.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 12 : i32} {
  tt.func @async_copy_mbarrier_arrive_partition_warp_mismatch(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    ttg.warp_specialize(%alloc) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(8) {
      // expected-error @+2 {{mbarrier arrivals require every arriving region to have ttg.num-warps (4) warps, but this region has 8}}
      // expected-error @+1 {{failed to legalize operation}}
      amdg.async_copy_mbarrier_arrive %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 6 : i32} {
  tt.func @tdm_load_barrier_partition_warp_mismatch(%desc: !tt.tensordesc<64x64xf16, #shared>, %dst: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %alloc: !ttg.memdesc<1xi64, #bar, #smem, mutable>) {
    ttg.warp_specialize(%desc, %dst, %alloc) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%d: !tt.tensordesc<64x64xf16, #shared>, %s: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<1xi64, #bar, #smem, mutable>) num_warps(2) {
      // expected-error @+2 {{mbarrier arrivals require every arriving region to have ttg.num-warps (4) warps, but this region has 2}}
      // expected-error @+1 {{failed to legalize operation}}
      %0 = amdg.async_tdm_copy_global_to_local %d into %s, barrier = %b : !tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!tt.tensordesc<64x64xf16, #shared>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>) -> ()
    tt.return
  }
}
