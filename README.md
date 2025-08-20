# hello-gemm


| Kernel | Performance (GFLOPS) | Relative to cuBLAS (%) |
| :--- | :--- | :--- |
| **cuBLAS** | 44978.2 | 100.0% |
| **CUTE_v1** | 29508.5 | 65.61% |
| **CUTE_v2** | 39739.9 | 88.35% |
| **CUTE_v3** | 36989.1 | 82.43% |

# Iterations
### Baseline: cuBLAS

## CUTE_v1 (65.61% baseline)
Most naive GEMM implemented from CuTE tutorial without any MMA or copy optimizations. Largely based on the example from [CuTe official doc](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#sgemm-1-cu)

## CUTE_v2 (88.35% baseline)
CUTE_v1 solution + Universal Copy + UniversalFMA 

### Ablation study - what made it faster?

TL;DR: it's a combination of SIMD inst (128-bit) and software pipelining.

- **SIMD instructions**
```
cuBLAS SGEMM: 44856.8GFLOPS

CUTE_v1 SGEMM:  29495 GFLOPS
CUTE_v1 / cuBLAS: 65.7536 %

CUTE_v2 (128-bit) SGEMM:  39840.9 GFLOPS
CUTE_v2 (128-bit) / cuBLAS: 88.818 %

CUTE_v2 (64-bit) SGEMM:  29749.6 GFLOPS
CUTE_v2 (64-bit) / cuBLAS: 66.3211 %

CUTE_v2 (32-bit) SGEMM:  28914.3 GFLOPS
CUTE_v2 (32-bit) / cuBLAS: 64.4591 %
```
As demonstrated above, when changing the copy instruction to 64-bit and 32-bit, the GFlops dropped to 65.7% and 63.9% respectively. 

Looking into PTX code, I can see different instructions are generated:
1. CUTE_v2 (32-bit)
```
	ld.global.v2.u64 	{%rd17, %rd18}, [%rd1];
	ld.global.v2.u64 	{%rd19, %rd20}, [%rd2];
	mov.b64 	{%r112, %r113}, %rd17;
	mov.b64 	{%r106, %r107}, %rd20;
	mov.b64 	{%r108, %r109}, %rd19;
	mov.b64 	{%r110, %r111}, %rd18;
    ...
    st.shared.v4.u32 	[%r13], {%r112, %r113, %r110, %r111};
	st.shared.v4.u32 	[%r12], {%r108, %r109, %r106, %r107};
```
2. CUTE_v2 (32-bit)
```
	ld.global.v2.u32 	{%r108, %r107}, [%rd2+8];
	ld.global.v2.u32 	{%r106, %r105}, [%rd2];
	ld.global.v2.u32 	{%r104, %r103}, [%rd1+8];
	ld.global.v2.u32 	{%r102, %r101}, [%rd1];
    ...
    st.shared.v4.u32 	[%r20], {%r102, %r101, %r104, %r103};
	st.shared.v4.u32 	[%r19], {%r106, %r105, %r108, %r107};
```
3. CUTE_v2 (32-bit)
```
	ld.global.u32 	%r95, [%rd2+12];
	ld.global.u32 	%r96, [%rd2+8];
	ld.global.u32 	%r93, [%rd2+4];
	ld.global.u32 	%r94, [%rd2];
	ld.global.u32 	%r91, [%rd1+12];
	ld.global.u32 	%r92, [%rd1+8];
	ld.global.u32 	%r89, [%rd1+4];
	ld.global.u32 	%r90, [%rd1];
    ...
	st.shared.v4.u32 	[%r17], {%r90, %r89, %r92, %r91};
	st.shared.v4.u32 	[%r16], {%r94, %r93, %r96, %r95};
```
4. CUTE_v1
```
	ld.global.f32 	%f324, [%rd16];
	st.shared.f32 	[%r12], %f324;
	ld.global.f32 	%f325, [%rd16+128];
	st.shared.f32 	[%r12+128], %f325;
	ld.global.f32 	%f326, [%rd16+256];
	st.shared.f32 	[%r12+256], %f326;
	ld.global.f32 	%f327, [%rd16+384];
	st.shared.f32 	[%r12+384], %f327;
	ld.global.f32 	%f328, [%rd20];
	st.shared.f32 	[%r13], %f328;
	ld.global.f32 	%f329, [%rd20+128];
	st.shared.f32 	[%r13+128], %f329;
	ld.global.f32 	%f330, [%rd20+256];
	st.shared.f32 	[%r13+256], %f330;
	ld.global.f32 	%f331, [%rd20+384];
	st.shared.f32 	[%r13+384], %f331;
```

- **Software Pipelining**

```
w/o software pipelining

CUTE_v1 SGEMM:  29491.2 GFLOPS
CUTE_v1 / cuBLAS: 65.7418 %

CUTE_v2 (128-bit) SGEMM:  29431 GFLOPS
CUTE_v2 (128-bit) / cuBLAS: 65.6077 %

CUTE_v2 (64-bit) SGEMM:  30574 GFLOPS
CUTE_v2 (64-bit) / cuBLAS: 68.1558 %

CUTE_v2 (32-bit) SGEMM:  28603.2 GFLOPS
CUTE_v2 (32-bit) / cuBLAS: 63.7624 %
```

## CUTE_v3 (82.43 % baseline)
This is essentially the sgemm_sm70.cu program in CuTe tutorial. I tested it on my H100 VM, and the additional software pipelining ends up making the perf worse.
```
CUTE_v3 SGEMM:  36986.3 GFLOPS
CUTE_v3 / cuBLAS: 82.427 %
```

That being said, one interesting observation is that the `CUTE_UNROLL` hint **significantly** affects the perf. When commenting out this hint, GFLOPS dropped to only a fraction of what it was:
```
CUTE_v3 SGEMM:  7821.13 GFLOPS
CUTE_v3 / cuBLAS: 17.4292 %
```

[WIP]