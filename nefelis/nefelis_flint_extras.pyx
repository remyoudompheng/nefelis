"""
Extra bindings for FLINT
"""

import cython
import flint

# No-GIL variants of polynomial multiplication functions.
# This is used for multithreading in the Block Wiedemann algorithm.

cdef extern from "flint/fmpz.h":
    ctypedef long slong

    # If FLINT was compiled in "no-reentrant" mode this may be necessary
    # to avoid a giant memory leak
    # FIXME: it doesn't seem to improve anything
    void _fmpz_cleanup() nogil

cdef extern from "flint/fmpz_mod_types.h":
    ctypedef struct fmpz_mod_ctx_struct:
        void * n
        # ...

    ctypedef fmpz_mod_ctx_struct fmpz_mod_ctx_t[1]

cdef extern from "flint/fmpz_mod_poly.h":
    ctypedef struct fmpz_mod_poly_struct:
        void * coeffs
        slong alloc
        slong length

    ctypedef fmpz_mod_poly_struct fmpz_mod_poly_t[1]

cdef extern from "flint/fmpz_mod_poly.h" nogil:
    void fmpz_mod_poly_init(fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)
    void fmpz_mod_poly_mul(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, const fmpz_mod_ctx_t ctx)
    void fmpz_mod_poly_mullow(fmpz_mod_poly_t res, const fmpz_mod_poly_t poly1, const fmpz_mod_poly_t poly2, slong n, const fmpz_mod_ctx_t ctx)

# HACK: minimal definitions to reproduce memory layout of python-flint

cdef class fmpz_mod_ctx:
    cdef fmpz_mod_ctx_t val
    # ...

    cdef method1(self):
        pass
    cdef method2(self):
        pass
    cdef method3(self):
        pass

cdef class fmpz_mod_poly_ctx:
    cdef fmpz_mod_ctx mod

    cdef method1(self):
        pass
    cdef method2(self):
        pass
    cdef method3(self):
        pass
    cdef method4(self):
        pass

cdef class fmpz_mod_poly:
    cdef fmpz_mod_poly_t val
    cdef fmpz_mod_poly_ctx ctx
    cpdef long length(self):
        raise
    cpdef long degree(self):
        raise

def nogil_fmpz_mod_poly_mul(f, g, ctx):
    assert isinstance(f, flint.fmpz_mod_poly)
    assert isinstance(g, flint.fmpz_mod_poly)

    res = flint.fmpz_mod_poly(0, ctx)
    cdef fmpz_mod_poly_struct *cres = &(<fmpz_mod_poly>res).val[0]
    cdef fmpz_mod_poly_struct *cf = &(<fmpz_mod_poly>f).val[0]
    cdef fmpz_mod_poly_struct *cg = &(<fmpz_mod_poly>g).val[0]
    cdef fmpz_mod_ctx_struct *cctx = &(<fmpz_mod_poly_ctx>ctx).mod.val[0]
    with cython.nogil:
        fmpz_mod_poly_mul(cres, cf, cg, cctx)
        # Clear cached allocations if necessary
        #_fmpz_cleanup()

    return res

def nogil_fmpz_mod_poly_mullow(f, g, slong b, ctx):
    assert isinstance(f, flint.fmpz_mod_poly)
    assert isinstance(g, flint.fmpz_mod_poly)

    res = flint.fmpz_mod_poly(0, ctx)
    cdef fmpz_mod_poly_struct *cres = &(<fmpz_mod_poly>res).val[0]
    cdef fmpz_mod_poly_struct *cf = &(<fmpz_mod_poly>f).val[0]
    cdef fmpz_mod_poly_struct *cg = &(<fmpz_mod_poly>g).val[0]
    cdef fmpz_mod_ctx_struct *cctx = &(<fmpz_mod_poly_ctx>ctx).mod.val[0]
    cdef slong bi = b
    with cython.nogil:
        fmpz_mod_poly_mullow(cres, cf, cg, b, cctx)
        # Clear cached allocations if necessary
        #_fmpz_cleanup()

    return res
