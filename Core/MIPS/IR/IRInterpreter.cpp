#include <algorithm>
#include <cmath>

#include "ppsspp_config.h"
#include "math/math_util.h"
#include "Common/Common.h"

#ifdef _M_SSE
#include <emmintrin.h>
#endif

#if PPSSPP_ARCH(ARM_NEON)
#include <arm_neon.h>
#endif

#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/Debugger/Breakpoints.h"
#include "Core/HLE/HLE.h"
#include "Core/HLE/ReplaceTables.h"
#include "Core/Host.h"
#include "Core/MemMap.h"
#include "Core/MIPS/MIPS.h"
#include "Core/MIPS/MIPSTables.h"
#include "Core/MIPS/MIPSVFPUUtils.h"
#include "Core/MIPS/IR/IRInst.h"
#include "Core/MIPS/IR/IRInterpreter.h"
#include "Core/System.h"

alignas(16) static const float vec4InitValues[8][4] = {
	{ 0.0f, 0.0f, 0.0f, 0.0f },
	{ 1.0f, 1.0f, 1.0f, 1.0f },
	{ -1.0f, -1.0f, -1.0f, -1.0f },
	{ 1.0f, 0.0f, 0.0f, 0.0f },
	{ 0.0f, 1.0f, 0.0f, 0.0f },
	{ 0.0f, 0.0f, 1.0f, 0.0f },
	{ 0.0f, 0.0f, 0.0f, 1.0f },
};

alignas(16) static const uint32_t signBits[4] = {
	0x80000000, 0x80000000, 0x80000000, 0x80000000,
};

alignas(16) static const uint32_t noSignMask[4] = {
	0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
};

alignas(16) static const uint32_t lowBytesMask[4] = {
	0x000000FF, 0x000000FF, 0x000000FF, 0x000000FF,
};

u32 RunBreakpoint(u32 pc) {
	// Should we skip this breakpoint?
	if (CBreakPoints::CheckSkipFirst() == pc)
		return 0;

	CBreakPoints::ExecBreakPoint(currentMIPS->pc);
	return coreState != CORE_RUNNING ? 1 : 0;
}

u32 RunMemCheck(u32 pc, u32 addr) {
	// Should we skip this breakpoint?
	if (CBreakPoints::CheckSkipFirst() == pc)
		return 0;

	CBreakPoints::ExecOpMemCheck(addr, pc);
	return coreState != CORE_RUNNING ? 1 : 0;
}

// We cannot use NEON on ARM32 here until we make it a hard dependency. We can, however, on ARM64.
u32 IRInterpret(MIPSState *Mips, const IRInst *inst, int count) {
	const IRInst *end = inst + count;
	while (inst != end) {
		switch (inst->op) {
		case IROp::Nop:
			_assert_(false);
			break;
		case IROp::SetConst:
			Mips->r[inst->dest] = inst->constant;
			break;
		case IROp::SetConstF:
			memcpy(&Mips->f[inst->dest], &inst->constant, 4);
			break;
		case IROp::Add:
			Mips->r[inst->dest] = Mips->r[inst->src1] + Mips->r[inst->src2];
			break;
		case IROp::Sub:
			Mips->r[inst->dest] = Mips->r[inst->src1] - Mips->r[inst->src2];
			break;
		case IROp::And:
			Mips->r[inst->dest] = Mips->r[inst->src1] & Mips->r[inst->src2];
			break;
		case IROp::Or:
			Mips->r[inst->dest] = Mips->r[inst->src1] | Mips->r[inst->src2];
			break;
		case IROp::Xor:
			Mips->r[inst->dest] = Mips->r[inst->src1] ^ Mips->r[inst->src2];
			break;
		case IROp::Mov:
			Mips->r[inst->dest] = Mips->r[inst->src1];
			break;
		case IROp::AddConst:
			Mips->r[inst->dest] = Mips->r[inst->src1] + inst->constant;
			break;
		case IROp::SubConst:
			Mips->r[inst->dest] = Mips->r[inst->src1] - inst->constant;
			break;
		case IROp::AndConst:
			Mips->r[inst->dest] = Mips->r[inst->src1] & inst->constant;
			break;
		case IROp::OrConst:
			Mips->r[inst->dest] = Mips->r[inst->src1] | inst->constant;
			break;
		case IROp::XorConst:
			Mips->r[inst->dest] = Mips->r[inst->src1] ^ inst->constant;
			break;
		case IROp::Neg:
			Mips->r[inst->dest] = -(s32)Mips->r[inst->src1];
			break;
		case IROp::Not:
			Mips->r[inst->dest] = ~Mips->r[inst->src1];
			break;
		case IROp::Ext8to32:
			Mips->r[inst->dest] = (s32)(s8)Mips->r[inst->src1];
			break;
		case IROp::Ext16to32:
			Mips->r[inst->dest] = (s32)(s16)Mips->r[inst->src1];
			break;
		case IROp::ReverseBits:
			Mips->r[inst->dest] = ReverseBits32(Mips->r[inst->src1]);
			break;

		case IROp::Load8:
			Mips->r[inst->dest] = Memory::ReadUnchecked_U8(Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Load8Ext:
			Mips->r[inst->dest] = (s32)(s8)Memory::ReadUnchecked_U8(Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Load16:
			Mips->r[inst->dest] = Memory::ReadUnchecked_U16(Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Load16Ext:
			Mips->r[inst->dest] = (s32)(s16)Memory::ReadUnchecked_U16(Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Load32:
			Mips->r[inst->dest] = Memory::ReadUnchecked_U32(Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Load32Left:
		{
			u32 addr = Mips->r[inst->src1] + inst->constant;
			u32 shift = (addr & 3) * 8;
			u32 mem = Memory::ReadUnchecked_U32(addr & 0xfffffffc);
			u32 destMask = 0x00ffffff >> shift;
			Mips->r[inst->dest] = (Mips->r[inst->dest] & destMask) | (mem << (24 - shift));
			break;
		}
		case IROp::Load32Right:
		{
			u32 addr = Mips->r[inst->src1] + inst->constant;
			u32 shift = (addr & 3) * 8;
			u32 mem = Memory::ReadUnchecked_U32(addr & 0xfffffffc);
			u32 destMask = 0xffffff00 << (24 - shift);
			Mips->r[inst->dest] = (Mips->r[inst->dest] & destMask) | (mem >> shift);
			break;
		}
		case IROp::LoadFloat:
			Mips->f[inst->dest] = Memory::ReadUnchecked_Float(Mips->r[inst->src1] + inst->constant);
			break;

		case IROp::Store8:
			Memory::WriteUnchecked_U8(Mips->r[inst->src3], Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Store16:
			Memory::WriteUnchecked_U16(Mips->r[inst->src3], Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Store32:
			Memory::WriteUnchecked_U32(Mips->r[inst->src3], Mips->r[inst->src1] + inst->constant);
			break;
		case IROp::Store32Left:
		{
			u32 addr = Mips->r[inst->src1] + inst->constant;
			u32 shift = (addr & 3) * 8;
			u32 mem = Memory::ReadUnchecked_U32(addr & 0xfffffffc);
			u32 memMask = 0xffffff00 << shift;
			u32 result = (Mips->r[inst->src3] >> (24 - shift)) | (mem & memMask);
			Memory::WriteUnchecked_U32(result, addr & 0xfffffffc);
			break;
		}
		case IROp::Store32Right:
		{
			u32 addr = Mips->r[inst->src1] + inst->constant;
			u32 shift = (addr & 3) * 8;
			u32 mem = Memory::ReadUnchecked_U32(addr & 0xfffffffc);
			u32 memMask = 0x00ffffff >> (24 - shift);
			u32 result = (Mips->r[inst->src3] << shift) | (mem & memMask);
			Memory::WriteUnchecked_U32(result, addr & 0xfffffffc);
			break;
		}
		case IROp::StoreFloat:
			Memory::WriteUnchecked_Float(Mips->f[inst->src3], Mips->r[inst->src1] + inst->constant);
			break;

		case IROp::LoadVec4:
		{
			u32 base = Mips->r[inst->src1] + inst->constant;
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_load_ps((const float *)Memory::GetPointerUnchecked(base)));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Memory::ReadUnchecked_Float(base + 4 * i);
#endif
			break;
		}
		case IROp::StoreVec4:
		{
			u32 base = Mips->r[inst->src1] + inst->constant;
#if defined(_M_SSE)
			_mm_store_ps((float *)Memory::GetPointerUnchecked(base), _mm_load_ps(&Mips->f[inst->dest]));
#else
			for (int i = 0; i < 4; i++)
				Memory::WriteUnchecked_Float(Mips->f[inst->dest + i], base + 4 * i);
#endif
			break;
		}

		case IROp::Vec4Init:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_load_ps(vec4InitValues[inst->src1]));
#else
			memcpy(&Mips->f[inst->dest], vec4InitValues[inst->src1], 4 * sizeof(float));
#endif
			break;
		}

		case IROp::Vec4Shuffle:
		{
			// Can't use the SSE shuffle here because it takes an immediate. pshufb with a table would work though,
			// or a big switch - there are only 256 shuffles possible (4^4)
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Mips->f[inst->src1 + ((inst->src2 >> (i * 2)) & 3)];
			break;
		}

		case IROp::Vec4Mov:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_load_ps(&Mips->f[inst->src1]));
#elif PPSSPP_ARCH(ARM64)
			vst1q_f32(&Mips->f[inst->dest], vld1q_f32(&Mips->f[inst->src1]));
#else
			memcpy(&Mips->f[inst->dest], &Mips->f[inst->src1], 4 * sizeof(float));
#endif
			break;
		}

		case IROp::Vec4Add:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_add_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_load_ps(&Mips->f[inst->src2])));
#elif PPSSPP_ARCH(ARM64)
			vst1q_f32(&Mips->f[inst->dest], vaddq_f32(vld1q_f32(&Mips->f[inst->src1]), vld1q_f32(&Mips->f[inst->src2])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Mips->f[inst->src1 + i] + Mips->f[inst->src2 + i];
#endif
			break;
		}

		case IROp::Vec4Sub:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_sub_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_load_ps(&Mips->f[inst->src2])));
#elif PPSSPP_ARCH(ARM64)
			vst1q_f32(&Mips->f[inst->dest], vsubq_f32(vld1q_f32(&Mips->f[inst->src1]), vld1q_f32(&Mips->f[inst->src2])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Mips->f[inst->src1 + i] - Mips->f[inst->src2 + i];
#endif
			break;
		}

		case IROp::Vec4Mul:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_mul_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_load_ps(&Mips->f[inst->src2])));
#elif PPSSPP_ARCH(ARM64)
			vst1q_f32(&Mips->f[inst->dest], vmulq_f32(vld1q_f32(&Mips->f[inst->src1]), vld1q_f32(&Mips->f[inst->src2])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Mips->f[inst->src1 + i] * Mips->f[inst->src2 + i];
#endif
			break;
		}

		case IROp::Vec4Div:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_div_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_load_ps(&Mips->f[inst->src2])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Mips->f[inst->src1 + i] / Mips->f[inst->src2 + i];
#endif
			break;
		}

		case IROp::Vec4Scale:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_mul_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_set1_ps(Mips->f[inst->src2])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = Mips->f[inst->src1 + i] * Mips->f[inst->src2];
#endif
			break;
		}

		case IROp::Vec4Neg:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_xor_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_load_ps((const float *)signBits)));
#elif PPSSPP_ARCH(ARM64)
			vst1q_f32(&Mips->f[inst->dest], vnegq_f32(vld1q_f32(&Mips->f[inst->src1])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = -Mips->f[inst->src1 + i];
#endif
			break;
		}

		case IROp::Vec4Abs:
		{
#if defined(_M_SSE)
			_mm_store_ps(&Mips->f[inst->dest], _mm_and_ps(_mm_load_ps(&Mips->f[inst->src1]), _mm_load_ps((const float *)noSignMask)));
#elif PPSSPP_ARCH(ARM64)
			vst1q_f32(&Mips->f[inst->dest], vabsq_f32(vld1q_f32(&Mips->f[inst->src1])));
#else
			for (int i = 0; i < 4; i++)
				Mips->f[inst->dest + i] = fabsf(Mips->f[inst->src1 + i]);
#endif
			break;
		}

		case IROp::Vec2Unpack16To31:
		{
			Mips->fi[inst->dest] = (Mips->fi[inst->src1] << 16) >> 1;
			Mips->fi[inst->dest + 1] = (Mips->fi[inst->src1] & 0xFFFF0000) >> 1;
			break;
		}

		case IROp::Vec2Unpack16To32:
		{
			Mips->fi[inst->dest] = (Mips->fi[inst->src1] << 16);
			Mips->fi[inst->dest + 1] = (Mips->fi[inst->src1] & 0xFFFF0000);
			break;
		}

		case IROp::Vec4Unpack8To32:
		{
#if defined(_M_SSE)
			__m128i src = _mm_cvtsi32_si128(Mips->fi[inst->src1]);
			src = _mm_unpacklo_epi8(src, _mm_setzero_si128());
			src = _mm_unpacklo_epi16(src, _mm_setzero_si128());
			_mm_store_si128((__m128i *)&Mips->fi[inst->dest], _mm_slli_epi32(src, 24));
#else
			Mips->fi[inst->dest] = (Mips->fi[inst->src1] << 24);
			Mips->fi[inst->dest + 1] = (Mips->fi[inst->src1] << 16) & 0xFF000000;
			Mips->fi[inst->dest + 2] = (Mips->fi[inst->src1] << 8) & 0xFF000000;
			Mips->fi[inst->dest + 3] = (Mips->fi[inst->src1]) & 0xFF000000;
#endif
			break;
		}

		case IROp::Vec2Pack32To16:
		{
			u32 val = Mips->fi[inst->src1] >> 16;
			Mips->fi[inst->dest] = (Mips->fi[inst->src1 + 1] & 0xFFFF0000) | val;
			break;
		}

		case IROp::Vec2Pack31To16:
		{
			u32 val = (Mips->fi[inst->src1] >> 15) & 0xFFFF;
			val |= (Mips->fi[inst->src1 + 1] << 1) & 0xFFFF0000;
			Mips->fi[inst->dest] = val;
			break;
		}

		case IROp::Vec4Pack32To8:
		{
			// Removed previous SSE code due to the need for unsigned 16-bit pack, which I'm too lazy to work around the lack of in SSE2.
			// pshufb or SSE4 instructions can be used instead.
			u32 val = Mips->fi[inst->src1] >> 24;
			val |= (Mips->fi[inst->src1 + 1] >> 16) & 0xFF00;
			val |= (Mips->fi[inst->src1 + 2] >> 8) & 0xFF0000;
			val |= (Mips->fi[inst->src1 + 3]) & 0xFF000000;
			Mips->fi[inst->dest] = val;
			break;
		}

		case IROp::Vec4Pack31To8:
		{
			// Removed previous SSE code due to the need for unsigned 16-bit pack, which I'm too lazy to work around the lack of in SSE2.
			// pshufb or SSE4 instructions can be used instead.
			u32 val = (Mips->fi[inst->src1] >> 23) & 0xFF;
			val |= (Mips->fi[inst->src1 + 1] >> 15) & 0xFF00;
			val |= (Mips->fi[inst->src1 + 2] >> 7) & 0xFF0000;
			val |= (Mips->fi[inst->src1 + 3] << 1) & 0xFF000000;
			Mips->fi[inst->dest] = val;
			break;
		}

		case IROp::Vec2ClampToZero:
		{
			for (int i = 0; i < 2; i++) {
				u32 val = Mips->fi[inst->src1 + i];
				Mips->fi[inst->dest + i] = (int)val >= 0 ? val : 0;
			}
			break;
		}

		case IROp::Vec4ClampToZero:
		{
#if defined(_M_SSE)
			// Trickery: Expand the sign bit, and use andnot to zero negative values.
			__m128i val = _mm_load_si128((const __m128i *)&Mips->fi[inst->src1]);
			__m128i mask = _mm_srai_epi32(val, 31);
			val = _mm_andnot_si128(mask, val);
			_mm_store_si128((__m128i *)&Mips->fi[inst->dest], val);
#else
			for (int i = 0; i < 4; i++) {
				u32 val = Mips->fi[inst->src1 + i];
				Mips->fi[inst->dest + i] = (int)val >= 0 ? val : 0;
			}
#endif
			break;
		}

		case IROp::Vec4DuplicateUpperBitsAndShift1:  // For vuc2i, the weird one.
		{
			for (int i = 0; i < 4; i++) {
				u32 val = Mips->fi[inst->src1 + i];
				val = val | (val >> 8);
				val = val | (val >> 16);
				val >>= 1;
				Mips->fi[inst->dest + i] = val;
			}
			break;
		}

		case IROp::FCmpVfpuBit:
		{
			int op = inst->dest & 0xF;
			int bit = inst->dest >> 4;
			int result = 0;
			switch (op) {
			case VC_EQ: result = Mips->f[inst->src1] == Mips->f[inst->src2]; break;
			case VC_NE: result = Mips->f[inst->src1] != Mips->f[inst->src2]; break;
			case VC_LT: result = Mips->f[inst->src1] < Mips->f[inst->src2]; break;
			case VC_LE: result = Mips->f[inst->src1] <= Mips->f[inst->src2]; break;
			case VC_GT: result = Mips->f[inst->src1] > Mips->f[inst->src2]; break;
			case VC_GE: result = Mips->f[inst->src1] >= Mips->f[inst->src2]; break;
			case VC_EZ: result = Mips->f[inst->src1] == 0.0f; break;
			case VC_NZ: result = Mips->f[inst->src1] != 0.0f; break;
			case VC_EN: result = my_isnan(Mips->f[inst->src1]); break;
			case VC_NN: result = !my_isnan(Mips->f[inst->src1]); break;
			case VC_EI: result = my_isinf(Mips->f[inst->src1]); break;
			case VC_NI: result = !my_isinf(Mips->f[inst->src1]); break;
			case VC_ES: result = my_isnanorinf(Mips->f[inst->src1]); break;
			case VC_NS: result = !my_isnanorinf(Mips->f[inst->src1]); break;
			case VC_TR: result = 1; break;
			case VC_FL: result = 0; break;
			default:
				result = 0;
			}
			if (result != 0) {
				Mips->vfpuCtrl[VFPU_CTRL_CC] |= (1 << bit);
			} else {
				Mips->vfpuCtrl[VFPU_CTRL_CC] &= ~(1 << bit);
			}
			break;
		}

		case IROp::FCmpVfpuAggregate:
		{
			u32 mask = inst->dest;
			u32 cc = Mips->vfpuCtrl[VFPU_CTRL_CC];
			int anyBit = (cc & mask) ? 0x10 : 0x00;
			int allBit = (cc & mask) == mask ? 0x20 : 0x00;
			Mips->vfpuCtrl[VFPU_CTRL_CC] = (cc & ~0x30) | anyBit | allBit;
			break;
		}

		case IROp::FCmovVfpuCC:
			if (((Mips->vfpuCtrl[VFPU_CTRL_CC] >> (inst->src2 & 0xf)) & 1) == ((u32)inst->src2 >> 7)) {
				Mips->f[inst->dest] = Mips->f[inst->src1];
			}
			break;

		// Not quickly implementable on all platforms, unfortunately.
		case IROp::Vec4Dot:
		{
			float dot = Mips->f[inst->src1] * Mips->f[inst->src2];
			for (int i = 1; i < 4; i++)
				dot += Mips->f[inst->src1 + i] * Mips->f[inst->src2 + i];
			Mips->f[inst->dest] = dot;
			break;
		}

		case IROp::FSin:
			Mips->f[inst->dest] = vfpu_sin(Mips->f[inst->src1]);
			break;
		case IROp::FCos:
			Mips->f[inst->dest] = vfpu_cos(Mips->f[inst->src1]);
			break;
		case IROp::FRSqrt:
			Mips->f[inst->dest] = 1.0f / sqrtf(Mips->f[inst->src1]);
			break;
		case IROp::FRecip:
			Mips->f[inst->dest] = 1.0f / Mips->f[inst->src1];
			break;
		case IROp::FAsin:
			Mips->f[inst->dest] = vfpu_asin(Mips->f[inst->src1]);
			break;

		case IROp::ShlImm:
			Mips->r[inst->dest] = Mips->r[inst->src1] << (int)inst->src2;
			break;
		case IROp::ShrImm:
			Mips->r[inst->dest] = Mips->r[inst->src1] >> (int)inst->src2;
			break;
		case IROp::SarImm:
			Mips->r[inst->dest] = (s32)Mips->r[inst->src1] >> (int)inst->src2;
			break;
		case IROp::RorImm:
		{
			u32 x = Mips->r[inst->src1];
			int sa = inst->src2;
			Mips->r[inst->dest] = (x >> sa) | (x << (32 - sa));
		}
		break;

		case IROp::Shl:
			Mips->r[inst->dest] = Mips->r[inst->src1] << (Mips->r[inst->src2] & 31);
			break;
		case IROp::Shr:
			Mips->r[inst->dest] = Mips->r[inst->src1] >> (Mips->r[inst->src2] & 31);
			break;
		case IROp::Sar:
			Mips->r[inst->dest] = (s32)Mips->r[inst->src1] >> (Mips->r[inst->src2] & 31);
			break;
		case IROp::Ror:
		{
			u32 x = Mips->r[inst->src1];
			int sa = Mips->r[inst->src2] & 31;
			Mips->r[inst->dest] = (x >> sa) | (x << (32 - sa));
			break;
		}

		case IROp::Clz:
		{
			int x = 31;
			int count = 0;
			int value = Mips->r[inst->src1];
			while (x >= 0 && !(value & (1 << x))) {
				count++;
				x--;
			}
			Mips->r[inst->dest] = count;
			break;
		}

		case IROp::Slt:
			Mips->r[inst->dest] = (s32)Mips->r[inst->src1] < (s32)Mips->r[inst->src2];
			break;

		case IROp::SltU:
			Mips->r[inst->dest] = Mips->r[inst->src1] < Mips->r[inst->src2];
			break;

		case IROp::SltConst:
			Mips->r[inst->dest] = (s32)Mips->r[inst->src1] < (s32)inst->constant;
			break;

		case IROp::SltUConst:
			Mips->r[inst->dest] = Mips->r[inst->src1] < inst->constant;
			break;

		case IROp::MovZ:
			if (Mips->r[inst->src1] == 0)
				Mips->r[inst->dest] = Mips->r[inst->src2];
			break;
		case IROp::MovNZ:
			if (Mips->r[inst->src1] != 0)
				Mips->r[inst->dest] = Mips->r[inst->src2];
			break;

		case IROp::Max:
			Mips->r[inst->dest] = (s32)Mips->r[inst->src1] > (s32)Mips->r[inst->src2] ? Mips->r[inst->src1] : Mips->r[inst->src2];
			break;
		case IROp::Min:
			Mips->r[inst->dest] = (s32)Mips->r[inst->src1] < (s32)Mips->r[inst->src2] ? Mips->r[inst->src1] : Mips->r[inst->src2];
			break;

		case IROp::MtLo:
			Mips->lo = Mips->r[inst->src1];
			break;
		case IROp::MtHi:
			Mips->hi = Mips->r[inst->src1];
			break;
		case IROp::MfLo:
			Mips->r[inst->dest] = Mips->lo;
			break;
		case IROp::MfHi:
			Mips->r[inst->dest] = Mips->hi;
			break;

		case IROp::Mult:
		{
			s64 result = (s64)(s32)Mips->r[inst->src1] * (s64)(s32)Mips->r[inst->src2];
			memcpy(&Mips->lo, &result, 8);
			break;
		}
		case IROp::MultU:
		{
			u64 result = (u64)Mips->r[inst->src1] * (u64)Mips->r[inst->src2];
			memcpy(&Mips->lo, &result, 8);
			break;
		}
		case IROp::Madd:
		{
			s64 result;
			memcpy(&result, &Mips->lo, 8);
			result += (s64)(s32)Mips->r[inst->src1] * (s64)(s32)Mips->r[inst->src2];
			memcpy(&Mips->lo, &result, 8);
			break;
		}
		case IROp::MaddU:
		{
			s64 result;
			memcpy(&result, &Mips->lo, 8);
			result += (u64)Mips->r[inst->src1] * (u64)Mips->r[inst->src2];
			memcpy(&Mips->lo, &result, 8);
			break;
		}
		case IROp::Msub:
		{
			s64 result;
			memcpy(&result, &Mips->lo, 8);
			result -= (s64)(s32)Mips->r[inst->src1] * (s64)(s32)Mips->r[inst->src2];
			memcpy(&Mips->lo, &result, 8);
			break;
		}
		case IROp::MsubU:
		{
			s64 result;
			memcpy(&result, &Mips->lo, 8);
			result -= (u64)Mips->r[inst->src1] * (u64)Mips->r[inst->src2];
			memcpy(&Mips->lo, &result, 8);
			break;
		}

		case IROp::Div:
		{
			s32 numerator = (s32)Mips->r[inst->src1];
			s32 denominator = (s32)Mips->r[inst->src2];
			if (numerator == (s32)0x80000000 && denominator == -1) {
				Mips->lo = 0x80000000;
				Mips->hi = -1;
			} else if (denominator != 0) {
				Mips->lo = (u32)(numerator / denominator);
				Mips->hi = (u32)(numerator % denominator);
			} else {
				Mips->lo = numerator < 0 ? 1 : -1;
				Mips->hi = numerator;
			}
			break;
		}
		case IROp::DivU:
		{
			u32 numerator = Mips->r[inst->src1];
			u32 denominator = Mips->r[inst->src2];
			if (denominator != 0) {
				Mips->lo = numerator / denominator;
				Mips->hi = numerator % denominator;
			} else {
				Mips->lo = numerator <= 0xFFFF ? 0xFFFF : -1;
				Mips->hi = numerator;
			}
			break;
		}

		case IROp::BSwap16:
		{
			u32 x = Mips->r[inst->src1];
			Mips->r[inst->dest] = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
			break;
		}
		case IROp::BSwap32:
		{
			u32 x = Mips->r[inst->src1];
			Mips->r[inst->dest] = ((x & 0xFF000000) >> 24) | ((x & 0x00FF0000) >> 8) | ((x & 0x0000FF00) << 8) | ((x & 0x000000FF) << 24);
			break;
		}

		case IROp::FAdd:
			Mips->f[inst->dest] = Mips->f[inst->src1] + Mips->f[inst->src2];
			break;
		case IROp::FSub:
			Mips->f[inst->dest] = Mips->f[inst->src1] - Mips->f[inst->src2];
			break;
		case IROp::FMul:
			Mips->f[inst->dest] = Mips->f[inst->src1] * Mips->f[inst->src2];
			break;
		case IROp::FDiv:
			Mips->f[inst->dest] = Mips->f[inst->src1] / Mips->f[inst->src2];
			break;
		case IROp::FMin:
			Mips->f[inst->dest] = std::min(Mips->f[inst->src1], Mips->f[inst->src2]);
			break;
		case IROp::FMax:
			Mips->f[inst->dest] = std::max(Mips->f[inst->src1], Mips->f[inst->src2]);
			break;

		case IROp::FMov:
			Mips->f[inst->dest] = Mips->f[inst->src1];
			break;
		case IROp::FAbs:
			Mips->f[inst->dest] = fabsf(Mips->f[inst->src1]);
			break;
		case IROp::FSqrt:
			Mips->f[inst->dest] = sqrtf(Mips->f[inst->src1]);
			break;
		case IROp::FNeg:
			Mips->f[inst->dest] = -Mips->f[inst->src1];
			break;
		case IROp::FSat0_1:
			// We have to do this carefully to handle NAN and -0.0f.
			Mips->f[inst->dest] = vfpu_clamp(Mips->f[inst->src1], 0.0f, 1.0f);
			break;
		case IROp::FSatMinus1_1:
			Mips->f[inst->dest] = vfpu_clamp(Mips->f[inst->src1], -1.0f, 1.0f);
			break;

		// Bitwise trickery
		case IROp::FSign:
		{
			u32 val;
			memcpy(&val, &Mips->f[inst->src1], sizeof(u32));
			if (val == 0 || val == 0x80000000)
				Mips->f[inst->dest] = 0.0f;
			else if ((val >> 31) == 0)
				Mips->f[inst->dest] = 1.0f;
			else
				Mips->f[inst->dest] = -1.0f;
			break;
		}

		case IROp::FpCondToReg:
			Mips->r[inst->dest] = Mips->fpcond;
			break;
		case IROp::VfpuCtrlToReg:
			Mips->r[inst->dest] = Mips->vfpuCtrl[inst->src1];
			break;
		case IROp::FRound:
		{
			float value = Mips->f[inst->src1];
			if (my_isnanorinf(value)) {
				Mips->fi[inst->dest] = my_isinf(value) && value < 0.0f ? -2147483648LL : 2147483647LL;
				break;
			} else {
				Mips->fs[inst->dest] = (int)floorf(value + 0.5f);
			}
			break;
		}
		case IROp::FTrunc:
		{
			float value = Mips->f[inst->src1];
			if (my_isnanorinf(value)) {
				Mips->fi[inst->dest] = my_isinf(value) && value < 0.0f ? -2147483648LL : 2147483647LL;
				break;
			} else {
				if (value >= 0.0f) {
					Mips->fs[inst->dest] = (int)floorf(value);
					// Overflow, but it was positive.
					if (Mips->fs[inst->dest] == -2147483648LL) {
						Mips->fs[inst->dest] = 2147483647LL;
					}
				} else {
					// Overflow happens to be the right value anyway.
					Mips->fs[inst->dest] = (int)ceilf(value);
				}
				break;
			}
		}
		case IROp::FCeil:
		{
			float value = Mips->f[inst->src1];
			if (my_isnanorinf(value)) {
				Mips->fi[inst->dest] = my_isinf(value) && value < 0.0f ? -2147483648LL : 2147483647LL;
				break;
			} else {
				Mips->fs[inst->dest] = (int)ceilf(value);
			}
			break;
		}
		case IROp::FFloor:
		{
			float value = Mips->f[inst->src1];
			if (my_isnanorinf(value)) {
				Mips->fi[inst->dest] = my_isinf(value) && value < 0.0f ? -2147483648LL : 2147483647LL;
				break;
			} else {
				Mips->fs[inst->dest] = (int)floorf(value);
			}
			break;
		}
		case IROp::FCmp:
			switch (inst->dest) {
			case IRFpCompareMode::False:
				Mips->fpcond = 0;
				break;
			case IRFpCompareMode::EitherUnordered:
			{
				float a = Mips->f[inst->src1];
				float b = Mips->f[inst->src2];
				Mips->fpcond = !(a > b || a < b || a == b);
				break;
			}
			case IRFpCompareMode::EqualOrdered:
			case IRFpCompareMode::EqualUnordered:
				Mips->fpcond = Mips->f[inst->src1] == Mips->f[inst->src2];
				break;
			case IRFpCompareMode::LessEqualOrdered:
			case IRFpCompareMode::LessEqualUnordered:
				Mips->fpcond = Mips->f[inst->src1] <= Mips->f[inst->src2];
				break;
			case IRFpCompareMode::LessOrdered:
			case IRFpCompareMode::LessUnordered:
				Mips->fpcond = Mips->f[inst->src1] < Mips->f[inst->src2];
				break;
			}
			break;

		case IROp::FCvtSW:
			Mips->f[inst->dest] = (float)Mips->fs[inst->src1];
			break;
		case IROp::FCvtWS:
		{
			float src = Mips->f[inst->src1];
			if (my_isnanorinf(src)) {
				Mips->fs[inst->dest] = my_isinf(src) && src < 0.0f ? -2147483648LL : 2147483647LL;
				break;
			}
			switch (Mips->fcr31 & 3) {
			case 0: Mips->fs[inst->dest] = (int)round_ieee_754(src); break;  // RINT_0
			case 1: Mips->fs[inst->dest] = (int)src; break;  // CAST_1
			case 2: Mips->fs[inst->dest] = (int)ceilf(src); break;  // CEIL_2
			case 3: Mips->fs[inst->dest] = (int)floorf(src); break;  // FLOOR_3
			}
			break; //cvt.w.s
		}

		case IROp::ZeroFpCond:
			Mips->fpcond = 0;
			break;

		case IROp::FMovFromGPR:
			memcpy(&Mips->f[inst->dest], &Mips->r[inst->src1], 4);
			break;
		case IROp::FMovToGPR:
			memcpy(&Mips->r[inst->dest], &Mips->f[inst->src1], 4);
			break;

		case IROp::ExitToConst:
			return inst->constant;

		case IROp::ExitToReg:
			return Mips->r[inst->src1];

		case IROp::ExitToConstIfEq:
			if (Mips->r[inst->src1] == Mips->r[inst->src2])
				return inst->constant;
			break;
		case IROp::ExitToConstIfNeq:
			if (Mips->r[inst->src1] != Mips->r[inst->src2])
				return inst->constant;
			break;
		case IROp::ExitToConstIfGtZ:
			if ((s32)Mips->r[inst->src1] > 0)
				return inst->constant;
			break;
		case IROp::ExitToConstIfGeZ:
			if ((s32)Mips->r[inst->src1] >= 0)
				return inst->constant;
			break;
		case IROp::ExitToConstIfLtZ:
			if ((s32)Mips->r[inst->src1] < 0)
				return inst->constant;
			break;
		case IROp::ExitToConstIfLeZ:
			if ((s32)Mips->r[inst->src1] <= 0)
				return inst->constant;
			break;

		case IROp::Downcount:
			Mips->downcount -= inst->constant;
			break;

		case IROp::SetPC:
			Mips->pc = Mips->r[inst->src1];
			break;

		case IROp::SetPCConst:
			Mips->pc = inst->constant;
			break;

		case IROp::Syscall:
			// IROp::SetPC was (hopefully) executed before.
		{
			MIPSOpcode op(inst->constant);
			CallSyscall(op);
			if (coreState != CORE_RUNNING)
				CoreTiming::ForceCheck();
			break;
		}

		case IROp::ExitToPC:
			return Mips->pc;

		case IROp::Interpret:  // SLOW fallback. Can be made faster. Ideally should be removed but may be useful for debugging.
		{
			MIPSOpcode op(inst->constant);
			MIPSInterpret(op);
			break;
		}

		case IROp::CallReplacement:
		{
			int funcIndex = inst->constant;
			const ReplacementTableEntry *f = GetReplacementFunc(funcIndex);
			int cycles = f->replaceFunc();
			Mips->downcount -= cycles;
			break;
		}

		case IROp::Break:
			if (!g_Config.bIgnoreBadMemAccess) {
				Core_EnableStepping(true);
				host->SetDebugMode(true);
			}
			return Mips->pc + 4;

		case IROp::SetCtrlVFPU:
			Mips->vfpuCtrl[inst->dest] = inst->constant;
			break;

		case IROp::SetCtrlVFPUReg:
			Mips->vfpuCtrl[inst->dest] = Mips->r[inst->src1];
			break;

		case IROp::SetCtrlVFPUFReg:
			memcpy(&Mips->vfpuCtrl[inst->dest], &Mips->f[inst->src1], 4);
			break;

		case IROp::Breakpoint:
			if (RunBreakpoint(Mips->pc)) {
				CoreTiming::ForceCheck();
				return Mips->pc;
			}
			break;

		case IROp::MemoryCheck:
			if (RunMemCheck(Mips->pc, Mips->r[inst->src1] + inst->constant)) {
				CoreTiming::ForceCheck();
				return Mips->pc;
			}
			break;

		case IROp::ApplyRoundingMode:
			// TODO: Implement
			break;
		case IROp::RestoreRoundingMode:
			// TODO: Implement
			break;
		case IROp::UpdateRoundingMode:
			// TODO: Implement
			break;

		default:
			// Unimplemented IR op. Bad.
			Crash();
		}
#ifdef _DEBUG
		if (Mips->r[0] != 0)
			Crash();
#endif
		inst++;
	}

	// If we got here, the block was badly constructed.
	Crash();
	return 0;
}
